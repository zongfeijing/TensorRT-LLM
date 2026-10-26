# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Expert DWDP across processes: every rank reads every expert, also on uneven partitions.

The test that runs the real expert transport across processes.  ``MPIPoolExecutor`` spawns
``NUM_RANKS`` ranks, each on GPU ``rank``.  Every rank enters a ``DwdpManager`` whose DWDP group
is the store rendezvous of context workers launched by plain ``trtllm-serve``
(``StoreDwdpGroup``: every collective times out after ``COLLECTIVE_TIMEOUT_S``), builds the same
toy model and runs ``DwdpManager.setup``, i.e. the real ``setup_dwdp``: the handshake, the
transport over FABRIC handles, ``WeightBuffer``, ``fill_edge_bytes``, ``DWDPWeightManager`` and
the fixup exchange.

The toy model has a layer without experts, then ``MOE_LAYERS``.  Their experts hold Kimi-K3's
four NVFP4 expert tensors with the first dimension cut (``WEIGHTS``): one weight larger than a
2 MiB page and three smaller ones, so a range's bounds fall inside pages and the handles carry
edge bytes of one or several neighbouring experts; no two elements of a layer's weight are
alike.  A rank's experts are the rows of its range in ``DwdpManager.expert_partition``, as
``MoE._init_dwdp_expert_layout`` allocates them.

``test_every_rank_reads_every_expert``: for each partition (balanced 8/7/7/7, capped 8/8/8/5,
and uniform and redundant as regression) and each transfer (per-slice copies, ``contention_opt``
and the SM gather), two forwards run back to back as the executor and ``ConfigurableMoE`` run
them: ``prefetch_first_layers``, then per MoE layer ``wait_and_bind``, a read of the bound
weights and ``record_compute_and_prefetch_next``, which prefetches the layer two ahead into the
slot just read.  The copy stream spins before each forward, longer than the compute stream spins
before each read, so a read that does not wait for its prefetch, or a prefetch that does not
wait for the read of the slot's previous layer, changes what a read sees.  Every expert of every
read must be the toy's, and ``fc31_alpha`` / ``fc2_alpha`` must be the full tables after the
fixup.

``test_one_rank_failure_raises_on_every_rank``: the last rank's prefetch ring does not fit (its
``torch.cuda.mem_get_info`` reports no free memory), after the transport.  Every rank must raise
within ``DEADLINE_S``, the last one its ``PrefetchRingMemoryError`` and the others a RuntimeError
naming it; a rank left waiting in a collective would raise only after ``COLLECTIVE_TIMEOUT_S``.
"""

from __future__ import annotations

import math
import os
import pickle
import sys
import time
from contextlib import nullcontext
from unittest.mock import patch

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from torch import nn

import tensorrt_llm
from tensorrt_llm._torch.modules.dwdp import DWDPWeightManager, fabric_gather, vmm
from tensorrt_llm._torch.pyexecutor.dwdp import DwdpManager
from tensorrt_llm._utils import get_free_ports, mpi_barrier, mpi_comm
from tensorrt_llm.llmapi.llm_args import DwdpConfig
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

# MPIPoolExecutor leaks a worker thread on first use; keep CI green.
pytestmark = pytest.mark.threadleak(enabled=False)

NUM_RANKS = 4
MOE_LAYERS = (1, 2, 3)
# Per-expert shape and dtype of the weights the transport moves: Kimi-K3's NVFP4 experts with the
# dtypes kept and the first dimension cut 2x for w3_w1_weight (5.25 MiB, larger than the real
# 2 MiB granularity, as K3's two weights are) and 8x for the others (smaller than a page, as K3's
# two scales are).
WEIGHTS = {
    "w3_w1_weight": ((3072, 224), torch.int64),
    "w2_weight": ((448, 192), torch.int64),
    "w3_w1_weight_scale": ((768, 56), torch.int32),
    "w2_weight_scale": ((448, 48), torch.int32),
}
ALPHAS = ("fc31_alpha", "fc2_alpha")
# (num_experts, num_experts_per_worker, num_prefetch_experts), and the ranges on NUM_RANKS ranks.
PARTITIONS = {
    "balanced": (29, 0, 0),  # [0, 8) [8, 15) [15, 22) [22, 29)
    "capped": (29, 8, 8),  # [0, 8) [8, 16) [16, 24) [24, 29)
    "uniform": (28, 7, 7),  # [0, 7) [7, 14) [14, 21) [21, 28)
    "redundant": (29, 11, 6),  # [0, 11) [6, 17) [12, 23) [18, 29)
}
TRANSFERS = ("per_slice", "contention_opt", "gather")
# Spins of the copy stream before a forward's first prefetch (about 4 ms) and of the compute
# stream before each read (about 1 ms); a toy layer's prefetch takes less than either.
PREFETCH_SPIN_CYCLES = 8_000_000
READ_SPIN_CYCLES = 2_000_000
COLLECTIVE_TIMEOUT_S = 120
DEADLINE_S = 60


# ---------------------------------------------------------------------------------------------
# One rank (runs in an MPIPoolExecutor worker)
# ---------------------------------------------------------------------------------------------


def _expected_weight(
    layer_idx: int, name: str, num_experts: int, device: torch.device
) -> torch.Tensor:
    """``[num_experts, *shape]`` of one weight of one MoE layer.

    No element equals another element of this layer's or another layer's weight.
    """
    shape, dtype = WEIGHTS[name]
    values = torch.arange(num_experts * math.prod(shape), device=device) * 4 + layer_idx
    return values.to(dtype).view(num_experts, *shape)


def _expected_alpha(
    layer_idx: int, name: str, num_experts: int, device: torch.device
) -> torch.Tensor:
    """``[num_experts]`` of one per-expert alpha of one MoE layer."""
    alpha = torch.arange(num_experts, dtype=torch.float32, device=device)
    return alpha + 100 * layer_idx + ALPHAS.index(name) / 2


class _ToyExperts(nn.Module):
    """One MoE layer's routed experts as ``setup_dwdp`` finds them: rows ``[start, end)``."""

    def __init__(
        self, layer_idx: int, num_experts: int, start: int, end: int, device: torch.device
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        for name in WEIGHTS:
            rows = _expected_weight(layer_idx, name, num_experts, device)[start:end]
            self.register_parameter(name, nn.Parameter(rows.clone(), requires_grad=False))
        for name in ALPHAS:
            rows = _expected_alpha(layer_idx, name, num_experts, device)[start:end]
            self.register_parameter(name, nn.Parameter(rows.clone(), requires_grad=False))
        # A per-layer activation scale: the exchange requires it to be equal on every rank.
        self.fc31_input_scale = nn.Parameter(torch.ones((), device=device), requires_grad=False)


class _ToyModel(nn.Module):
    """A layer without experts, then ``MOE_LAYERS``, whose ``mlp`` are ``_ToyExperts``."""

    def __init__(self, manager: DwdpManager, num_experts: int, device: torch.device) -> None:
        super().__init__()
        start, end = manager.local_expert_range(num_experts)
        self.layers = nn.ModuleList([nn.Module()])
        for layer_idx in MOE_LAYERS:
            manager.add_layer(layer_idx)
            layer = nn.Module()
            layer.mlp = _ToyExperts(layer_idx, num_experts, start, end, device)
            self.layers.append(layer)


def _forward(
    manager: DwdpManager, model: _ToyModel, copy_stream: torch.cuda.Stream
) -> dict[tuple[int, str], torch.Tensor]:
    """One forward's reads of the bound expert weights, by ``(layer_idx, name)``."""
    with torch.cuda.stream(copy_stream):
        torch.cuda._sleep(PREFETCH_SPIN_CYCLES)
    manager.prefetch_first_layers()
    reads = {}
    for layer_idx in MOE_LAYERS:
        experts = model.layers[layer_idx].mlp
        manager.wait_and_bind(experts, layer_idx)
        torch.cuda._sleep(READ_SPIN_CYCLES)
        for name in WEIGHTS:
            reads[layer_idx, name] = getattr(experts, name).clone()
        manager.record_compute_and_prefetch_next(layer_idx)
    return reads


def _check(
    manager: DwdpManager,
    model: _ToyModel,
    weight_manager: DWDPWeightManager,
    transfer: str,
    num_experts: int,
) -> list[str]:
    """What two forwards and the fixed-up alphas show that is wrong."""
    device = torch.device("cuda", torch.cuda.current_device())
    problems = []
    used = (weight_manager.use_gather, weight_manager.use_batched_prefetch)
    if used != (transfer == "gather", transfer == "contention_opt"):
        problems.append(f"(use_gather, use_batched_prefetch) is {used} for {transfer}")
    forwards = [_forward(manager, model, weight_manager.copy_stream) for _ in range(2)]
    torch.cuda.synchronize(device)
    for forward, reads in enumerate(forwards):
        for (layer_idx, name), read in reads.items():
            expected = _expected_weight(layer_idx, name, num_experts, device)
            wrong = (read != expected).flatten(1).any(1).nonzero().flatten().tolist()
            if wrong:
                problems.append(f"forward {forward}: layer {layer_idx} {name} experts {wrong}")
    for layer_idx in MOE_LAYERS:
        for name in ALPHAS:
            alpha = getattr(model.layers[layer_idx].mlp, name)
            if not torch.equal(alpha, _expected_alpha(layer_idx, name, num_experts, device)):
                problems.append(f"layer {layer_idx} {name} after the fixup: {alpha.tolist()}")
    return problems


def _rank_main(
    partition: str, transfer: str, fault_rank: int | None
) -> tuple[int, list[str], str | None, float]:
    """One DWDP rank on GPU ``rank``.

    Returns:
        ``(rank, problems, error, seconds)``: what ``_check`` found, the error ``setup`` raised
        (``"<type>: <message>"``, None if it returned) and the seconds it took.
    """
    rank = tensorrt_llm.mpi_rank()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    # Rank 0 serves the store; it picks the port once the pool's own sockets exist.
    port = mpi_comm().bcast(get_free_ports(1)[0] if rank == 0 else None)
    for name in [name for name in os.environ if name.startswith("TRTLLM_DWDP_")]:
        del os.environ[name]
    os.environ.update(
        TRTLLM_DWDP_RANK=str(rank),
        TRTLLM_DWDP_MASTER_ADDR="127.0.0.1",
        TRTLLM_DWDP_MASTER_PORT=str(port),
        TRTLLM_DWDP_RENDEZVOUS_TIMEOUT_S=str(COLLECTIVE_TIMEOUT_S),
        TRTLLM_DWDP_PREFETCH_GATHER="1" if transfer == "gather" else "0",
    )
    num_experts, size, stride = PARTITIONS[partition]
    config = DwdpConfig(
        dwdp_size=NUM_RANKS,
        num_experts_per_worker=size,
        num_prefetch_experts=stride,
        contention_opt=transfer == "contention_opt",
    )
    problems, error = [], None
    # A context worker's mapping: TP 1, the DWDP rank injected; the store needs no MPIDist.
    with DwdpManager(config, None, Mapping(dwdp_size=NUM_RANKS, dwdp_rank=rank)) as manager:
        model = _ToyModel(manager, num_experts, device)
        no_free_memory = patch.object(
            torch.cuda, "mem_get_info", return_value=(0, torch.cuda.mem_get_info(device)[1])
        )
        mpi_barrier()  # every rank starts setup at once: the time to an error is the setup's
        start = time.monotonic()
        try:
            with no_free_memory if rank == fault_rank else nullcontext():
                weight_manager = manager.setup(model)
        except Exception as exc:  # reported to the parent once every rank is past the barrier
            error = f"{type(exc).__name__}: {exc}"
        seconds = time.monotonic() - start
        # Rank 0 serves the group's store: no rank frees it before every rank has its results.
        mpi_barrier()
        if error is None:
            problems = _check(manager, model, weight_manager, transfer, num_experts)
    return rank, problems, error, seconds


# ---------------------------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------------------------


def _run(
    partition: str, transfer: str, fault_rank: int | None = None
) -> list[tuple[int, list[str], str | None, float]]:
    """Run ``NUM_RANKS`` ranks; their ``_rank_main`` reports, by rank."""
    if torch.cuda.device_count() < NUM_RANKS:
        pytest.skip(f"needs {NUM_RANKS} GPUs, found {torch.cuda.device_count()}")
    if vmm.peer_handle_type() != vmm.cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC:
        pytest.skip("expert DWDP across processes is tested with FABRIC peer handles (aarch64)")
    try:
        vmm.release_handle(vmm.create_fabric_handle(vmm.get_allocation_granularity(0), 0))
    except RuntimeError as exc:
        pytest.skip(f"cannot create a FABRIC handle: {exc}")
    if transfer == "gather" and fabric_gather.gather_op() is None:
        pytest.skip("trtllm::dwdp_fabric_gather is not registered by this build")
    with MPIPoolExecutor(max_workers=NUM_RANKS) as executor:
        futures = [
            executor.submit(_rank_main, partition, transfer, fault_rank) for _ in range(NUM_RANKS)
        ]
        return sorted(future.result() for future in futures)


@pytest.mark.parametrize("transfer", TRANSFERS)
@pytest.mark.parametrize("partition", PARTITIONS)
def test_every_rank_reads_every_expert(partition: str, transfer: str) -> None:
    """Every expert of every layer reaches every rank's composite intact, and the full alphas."""
    findings = [
        f"rank {rank}: {finding}"
        for rank, problems, error, _ in _run(partition, transfer)
        for finding in ([f"setup raised {error}"] if error else problems)
    ]
    assert not findings, "\n".join(findings)


def test_one_rank_failure_raises_on_every_rank() -> None:
    """The last rank's prefetch ring does not fit; every rank raises, at once."""
    fault_rank = NUM_RANKS - 1
    for rank, _, error, seconds in _run("balanced", "per_slice", fault_rank):
        if rank == fault_rank:
            assert error and error.startswith("PrefetchRingMemoryError: "), f"rank {rank}: {error}"
        else:
            named = f"rank {fault_rank}: PrefetchRingMemoryError"
            assert error and error.startswith("RuntimeError: ") and named in error, (
                f"rank {rank}: {error}"
            )
        assert seconds < DEADLINE_S, f"rank {rank} raised after {seconds:.0f} s"
