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
"""CPU tests of the expert DWDP setup protocol: ``setup_dwdp`` on a group of rank threads.

Every DWDP rank is a thread that runs the real ``setup_dwdp`` (handshake, transport, exchange of
the small expert parameters) on a toy model holding the rows of its experts.  The threads gather
over ``StoreDwdpGroup`` on one in-process ``HashStore`` whose waits are bounded, so a rank that
skips a collective fails the test instead of hanging it.  ``_FakeDevice`` runs the transport's
VMM calls on host memory; the weight buffer, edge-byte fill and weight manager are stand-ins.
"""

import math
import re
import threading
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributed import HashStore

from tensorrt_llm._torch.modules.dwdp import setup as dwdp_setup
from tensorrt_llm._torch.modules.dwdp import transport
from tensorrt_llm._torch.modules.dwdp.rendezvous import StoreDwdpGroup
from tensorrt_llm._torch.modules.dwdp.specs import ExpertPartition
from tensorrt_llm._torch.modules.dwdp.weight_buffer import PrefetchRingMemoryError

pytestmark = pytest.mark.cpu_only

#: Seconds one collective waits for the slowest rank; a rank that skips a collective times out.
COLLECTIVE_TIMEOUT_S = 20
#: Seconds a test waits for its rank threads.
JOIN_TIMEOUT_S = 60
GRANULARITY = 4096
LAYER = 1
#: int32 columns of the toy expert weights: 160 and 384 bytes per expert, so most ranges start
#: and end inside a page.
WEIGHTS = {"w2_weight": 40, "w3_w1_weight": 96}
ALL_COLLECTIVES = ["handshake", "export", "export", "import status", "vote"]

_RANK = threading.local()


def _rows(start, end, width):
    """Experts ``[start, end)`` of a toy weight: each row holds its expert's id."""
    return torch.arange(start, end, dtype=torch.int32)[:, None].repeat(1, width)


def _pages(start, end, expert_bytes):
    """Bytes of the pages experts ``[start, end)`` of one weight touch: that rank's handle size."""
    page_end = -(-end * expert_bytes // GRANULARITY) * GRANULARITY
    return page_end - start * expert_bytes // GRANULARITY * GRANULARITY


class _Experts(nn.Module):
    """A rank's MoE backend after loading: the rows of experts ``[start, end)``.

    ``weights`` replaces some of the toy weights, by name.
    """

    def __init__(self, num_experts, start, end, weights):
        super().__init__()
        self.num_experts = num_experts
        for name, width in WEIGHTS.items():
            rows = weights.get(name, _rows(start, end, width))
            setattr(self, name, nn.Parameter(rows, requires_grad=False))
        alpha = torch.arange(start, end, dtype=torch.float32)
        self.fc31_alpha = nn.Parameter(alpha, requires_grad=False)
        self.fc31_input_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)


class _Group(StoreDwdpGroup):
    """A rank's DWDP group on the test's store; it logs which collective each allgather is."""

    def __init__(self, store, rank, size):
        super().__init__(rank, size, "unused", timeout_s=COLLECTIVE_TIMEOUT_S, store=store)
        self.log = []

    def allgather(self, obj):
        if not isinstance(obj, tuple):
            self.log.append("import status")
        elif len(obj) == 6:
            self.log.append("handshake")
        else:  # (status, payload) of an export, (status, small parameters) of the vote
            self.log.append("vote" if isinstance(obj[1], dict) else "export")
        return super().allgather(obj)


class _WeightManager:
    """Stands in for ``DWDPWeightManager``: ``setup_dwdp`` hands it the transport to release."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._transport = None

    def release(self):
        self._transport.release()


class _FakeDevice:
    """The transport's VMM calls over host memory, shared by the rank threads.

    A handle is a ``uint8`` tensor, which its imports share, and a VA is a number.  As on the
    device, a handle maps only whole, so an importer must map a peer's handle with the size the
    peer allocated.  ``fault = (rank, step)``: that rank raises ``injected`` at its second handle
    (``create``), at the access grant of the first peer handle it imported and mapped
    (``import``) or at its prefetch ring (``ring``), or reports twice the granularity
    (``granularity``).
    """

    def __init__(self, monkeypatch):
        self.fault = self.injected = None
        self.handles, self.va_ranges, self.mappings = {}, {}, {}
        self.imported = set()
        self.calls = Counter()  # by step and by (rank, step)
        self._lock = threading.Lock()
        self._next_id = GRANULARITY
        fabric = transport.cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        fakes = {
            "get_allocation_granularity": self.granularity,
            "peer_handle_type": lambda: fabric,
            "create_fabric_handle": self.create,
            "_export_handle": lambda handle, htype: handle,
            "_import_handle": self.import_handle,
            "reserve_va": lambda size, alignment: self._add(self.va_ranges, size, size + alignment),
            "map_handle": self.map,
            "set_access": self.set_access,
            "tensor_from_ptr": self.view,
            "unmap_va": lambda va, size: self._drop(self.mappings, va),
            "free_va": lambda va, size: self._drop(self.va_ranges, va),
            "release_handle": lambda handle: self._drop(self.handles, handle),
        }
        for name, fake in fakes.items():
            monkeypatch.setattr(transport, name, fake)
        monkeypatch.setattr(dwdp_setup, "get_allocation_granularity", self.granularity)
        monkeypatch.setattr(dwdp_setup.WeightBuffer, "create", self.create_weight_buffer)
        monkeypatch.setattr(dwdp_setup, "fill_edge_bytes", lambda **_kwargs: None)
        monkeypatch.setattr(dwdp_setup, "DWDPWeightManager", _WeightManager)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (1 << 40, 1 << 41))

    def _step(self, step, failing_call=1):
        with self._lock:
            self.calls[step] += 1
            self.calls[_RANK.value, step] += 1
            count = self.calls[_RANK.value, step]
        if self.fault == (_RANK.value, step) and count == failing_call:
            raise self.injected

    def _add(self, table, value, span=1):
        with self._lock:
            key = self._next_id
            self._next_id += span
            table[key] = value
        return key

    def _drop(self, table, key):
        with self._lock:
            del table[key]

    def granularity(self, device_id, use_cache=True):
        return GRANULARITY * (2 if self.fault == (_RANK.value, "granularity") else 1)

    def create(self, size, device_id):
        self._step("create", failing_call=2)
        return self._add(self.handles, torch.zeros(size, dtype=torch.uint8))

    def import_handle(self, payload, htype):
        handle = self._add(self.handles, self.handles[payload])
        self.imported.add(handle)
        return handle

    def map(self, va, size, handle, offset=0):
        if offset or size != self.handles[handle].numel():
            raise RuntimeError("CUDA_ERROR_NOT_SUPPORTED: a handle maps only whole")
        with self._lock:
            self.mappings[va] = handle

    def set_access(self, va, size, device_id):
        if self.mappings[va] in self.imported:
            self._step("import")

    def view(self, ptr, shape, dtype, device_id):
        with self._lock:
            va = max(mapped for mapped in self.mappings if mapped <= ptr)
            memory = self.handles[self.mappings[va]]
        start, nbytes = ptr - va, math.prod(shape) * dtype.itemsize
        assert start + nbytes <= memory.numel(), "a view past the end of its mapping"
        return memory[start : start + nbytes].view(dtype).reshape(shape)

    def create_weight_buffer(self, **kwargs):
        self._step("ring")
        return SimpleNamespace(release=lambda: None, kwargs=kwargs)


@pytest.fixture
def device(monkeypatch):
    return _FakeDevice(monkeypatch)


def _run_setup(num_experts, configs, loaded, weights=None):
    """Run ``setup_dwdp`` on one thread per DWDP rank and return each rank's outcome.

    Rank ``r`` runs ``configs[r]`` (``num_experts_per_worker``, ``num_prefetch_experts``) on a
    model holding the rows of experts ``loaded[r]`` of ``num_experts``; ``weights[r]`` replaces
    some of its weights (``_Experts``).
    """
    weights = weights or {}
    store = HashStore()
    size = len(configs)
    outcomes = [SimpleNamespace(result=None, error=None, log=None, model=None) for _ in configs]

    def run(rank):
        _RANK.value = rank
        outcome = outcomes[rank]
        group = _Group(store, rank, size)
        outcome.log = group.log
        outcome.model = nn.Module()
        outcome.model.layers = nn.ModuleList([nn.Module(), nn.Module()])
        outcome.model.layers[LAYER].mlp = _Experts(
            num_experts, *loaded[rank], weights.get(rank, {})
        )
        try:
            outcome.result = dwdp_setup.setup_dwdp(
                model=outcome.model,
                mapping=SimpleNamespace(dwdp_enabled=True, dwdp_rank=rank, dwdp_size=size),
                device_id=0,
                comm=group,
                layer_indices=[LAYER],
                num_experts_per_worker=configs[rank][0],
                num_prefetch_experts=configs[rank][1],
            )
        except Exception as exc:  # the test inspects every rank's outcome
            outcome.error = exc

    threads = [threading.Thread(target=run, args=(rank,), daemon=True) for rank in range(size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(JOIN_TIMEOUT_S)
    stuck = [rank for rank, thread in enumerate(threads) if thread.is_alive()]
    assert stuck == [], f"setup_dwdp did not finish on ranks {stuck}"
    return outcomes


#: kind: (num_experts, dwdp_size, (num_experts_per_worker, num_prefetch_experts))
PARTITIONS = {
    "balanced": (896, 12, (0, 0)),  # 8 x 75 + 4 x 74
    "capped": (72, 2, (40, 40)),  # 40 + 32
    "uniform": (64, 4, (16, 16)),
    "redundant": (256, 3, (86, 85)),  # one expert on two ranks at each boundary
}


@pytest.mark.parametrize("kind", PARTITIONS)
def test_every_peer_is_mapped_whole_with_its_own_size_and_rows(device, kind):
    """A rank's handle spans the pages of its range; every peer maps it whole and views its rows.

    A peer's handle mapped with another size fails on the fake device; a wrong data offset or
    row count shows in the views.  Every rank makes the same collectives and gets the full
    ``fc31_alpha``.  Uniform and redundant ranges are ``[r * stride, r * stride + size)``.
    """
    num_experts, dwdp_size, (size, stride) = PARTITIONS[kind]
    partition = ExpertPartition.from_config(
        num_experts=num_experts,
        dwdp_size=dwdp_size,
        num_experts_per_worker=size,
        num_prefetch_experts=stride,
    )
    assert partition.kind == kind
    if kind in ("uniform", "redundant"):
        assert partition.peer_ranges == [(r * stride, r * stride + size) for r in range(dwdp_size)]

    outcomes = _run_setup(num_experts, [(size, stride)] * dwdp_size, partition.ranges)

    for rank, outcome in enumerate(outcomes):
        assert outcome.error is None, f"rank {rank}: {outcome.error!r}"
        assert outcome.log == ALL_COLLECTIVES
        handles = outcome.result._transport.get_handle_set()
        views = outcome.result._transport.get_peer_views()
        for name, width in WEIGHTS.items():
            assert handles.get_size(LAYER, name) == _pages(*partition.range_of(rank), 4 * width)
            for peer, (start, end) in enumerate(partition.ranges):
                if peer != rank:
                    assert torch.equal(views[(peer, LAYER, name)], _rows(start, end, width))
        manager = outcome.result.kwargs  # steps 5-7 get the partition's ranges
        assert manager["peer_ranges"] == partition.peer_ranges
        ring = manager["weight_buffer"].kwargs
        assert (ring["local_start"], ring["local_end"]) == partition.range_of(rank)
        alpha = outcome.model.layers[LAYER].mlp.fc31_alpha
        assert torch.equal(alpha, torch.arange(num_experts, dtype=torch.float32))
    for outcome in outcomes:
        outcome.result.release()
    assert device.handles == device.va_ranges == device.mappings == {}


FAILING = 2


@pytest.mark.parametrize(
    "fault, collectives",
    [
        ("rows", ["handshake"]),
        ("partition", ["handshake"]),
        ("granularity", ["handshake"]),
        ("shape", ["handshake"]),
        ("dtype", ["handshake"]),
        ("create", ["handshake", "export", "export"]),
        ("import", ALL_COLLECTIVES[:4]),
        ("ring", ALL_COLLECTIVES),
    ],
)
def test_a_failure_on_one_rank_raises_on_every_rank_after_the_same_collective(
    device, fault, collectives
):
    """Rank 2 fails; every rank raises after the collective that reports it and releases all.

    A handshake mismatch (rank 2 loaded a row more, computed another partition, maps with
    another granularity, or loaded experts of another shape or dtype than its peers size and
    view its handles with) raises the same error everywhere before any handle exists; after a
    failed export nothing is imported.  A rank whose own step failed raises its error, the
    others an error naming it.
    """
    partition = ExpertPartition.balanced(num_experts=13, dwdp_size=4)  # 4 + 3 + 3 + 3
    configs = [(0, 0)] * partition.dwdp_size
    loaded = list(partition.ranges)
    weights = {}
    if fault == "rows":
        loaded[FAILING] = (loaded[FAILING][0], loaded[FAILING][1] + 1)
    elif fault == "partition":
        configs[FAILING] = (4, 3)  # redundant: rank 2 stores [6, 10)
        loaded[FAILING] = (6, 10)
    elif fault == "shape":  # a column less per expert
        weights = {"w3_w1_weight": _rows(*loaded[FAILING], WEIGHTS["w3_w1_weight"] - 1)}
    elif fault == "dtype":  # the same bytes per expert
        weights = {"w2_weight": _rows(*loaded[FAILING], WEIGHTS["w2_weight"]).view(torch.float32)}
    else:
        device.fault = (FAILING, fault)
        device.injected = (PrefetchRingMemoryError if fault == "ring" else RuntimeError)("injected")

    outcomes = _run_setup(13, configs, loaded, {FAILING: weights})

    assert [outcome.log for outcome in outcomes] == [collectives] * partition.dwdp_size
    assert device.handles == device.va_ranges == device.mappings == {}
    if collectives == ["handshake"]:
        assert device.calls["create"] == 0
        assert all(type(outcome.error) is RuntimeError for outcome in outcomes)
        assert len({str(outcome.error) for outcome in outcomes}) == 1
        return
    if fault == "create":
        assert not device.imported
    for rank, outcome in enumerate(outcomes):
        if rank == FAILING:
            assert outcome.error is device.injected
        else:
            # A plain RuntimeError: a store timeout would be a DistStoreError.
            assert type(outcome.error) is RuntimeError, repr(outcome.error)
            assert re.search(rf"\brank {FAILING}\b", str(outcome.error)), str(outcome.error)


def test_the_handshake_names_rank_0_when_it_alone_computed_another_partition(device):
    """Each rank is compared with the most common report, so the error names rank 0 here."""
    loaded = ExpertPartition.balanced(num_experts=13, dwdp_size=4).ranges  # rank 0: [0, 4) too
    outcomes = _run_setup(13, [(4, 3)] + [(0, 0)] * 3, loaded)  # rank 0 redundant, 1-3 balanced

    assert all(type(outcome.error) is RuntimeError for outcome in outcomes)
    assert len({str(outcome.error) for outcome in outcomes}) == 1
    named = re.findall(r"\brank (\d+) computed the (\w+) partition", str(outcomes[0].error))
    assert named == [("0", "redundant")]
