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
"""The closed protocol of dense DWDP on one GPU: CUDA graphs and eager forwards stay bit-exact.

A ``DenseDwdpManager`` in graph mode sets dense DWDP up on a toy causal LM, standalone, as rank 0
of a 16-rank loopback group (the helpers of ``test_dense_k3_modules_gpu``), and commits the
``CLOSED`` protocol, with either transfer: the TMA SM gather (``TRTLLM_DWDP_DENSE_COPY=gather``,
with the gather's SM reserve, as on Rubin) or one batched copy-engine copy per layer
(``batched``, without an SM reserve, as on GB300).  The model has Kimi-K3's 93 decoder layers,
layer 0 uncovered, so the slots line up as Kimi-K3's do.  A covered layer reads ``attn``
(attention segment) between H1 and H2, then ``moe`` and ``shared`` (MoE-dense segment) between H2
and H3; under multi-stream execution ``shared`` runs on an aux stream that forks after H2 and
joins before H3, as the shared expert's does.  A real ``CUDAGraphRunner`` captures and replays the
forward, behind the replay guard the model registers as its pre-replay hook.  Spin kernels delay
chosen copies, read windows, aux reads and the tail, so that a missing stream edge shows up as
other bytes in an output; every output is compared with the reference model, which keeps the
replicated weights.  The gate's capture probe of ``cudaMemcpyBatchAsync`` passes on this GPU's
driver, and reports a failure with the CUDA versions.
"""

from __future__ import annotations

import contextlib
import copy
import os
import random
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import NamedTuple

import pytest
import torch
from _torch.helpers import create_mock_cuda_graph_runner
from test_dense_k3_modules_gpu import (
    _DEPLOYMENT_ENV,
    _NUM_RANKS,
    _dep_mapping,
    _LoopbackGroup,
    _require_gather,
    _spin,
)
from torch import nn

from tensorrt_llm._torch.modules.dwdp.dense import plan as dense_plan
from tensorrt_llm._torch.modules.dwdp.dense.config import DenseConfig
from tensorrt_llm._torch.modules.dwdp.dense.manager import DenseDwdpManager
from tensorrt_llm._torch.modules.dwdp.dense.plan import DenseCopyLauncher
from tensorrt_llm._torch.modules.dwdp.dense.runtime import DenseDWDP
from tensorrt_llm._torch.modules.dwdp.dense.store import LoopbackShardStore
from tensorrt_llm._torch.modules.dwdp.dense.types import DenseMode, ForwardState, PrimePolicy
from tensorrt_llm._torch.modules.multi_stream_utils import do_multi_stream, with_multi_stream
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner, KeyType

#: Decoder layers, Kimi-K3's 93: layer 0 stays replicated, layers 1..92 are covered.
_NUM_LAYERS = 93
_LAST = _NUM_LAYERS - 1
#: The middle layer, where one abort happens.
_MID = 45
_HIDDEN = 1024
_VOCAB = 64
#: GPU clock cycles of a spin kernel: far longer than a copy or a layer of the toy model.
_SPIN = 2_000_000

#: The transfers: the SM gather and the batched copy-engine copies.
_TRANSFERS = ("gather", "batched")

#: The gather's environment: S7's, but for the Kimi-K3 model switch (the TMA kernel, and the
#: compute-side settings that setup would otherwise set: the SM reserve, the GEMM scheduler and
#: non-cooperative routing).
_GATHER_ENV = {
    **{
        name: value
        for name, value in _DEPLOYMENT_ENV.items()
        if name != "KIMI_K3_FP8_WEIGHT_READ_MOE_MLP"
    },
    "TLLM_ROUTING_DISABLE_COOP_LAUNCH": "1",
}


class _Delays(NamedTuple):
    """The layers whose gather, read window (after H1) or aux-stream read starts with a spin."""

    copies: frozenset[int] = frozenset()
    reads: frozenset[int] = frozenset()
    aux: frozenset[int] = frozenset()
    tail: bool = False
    """Whether the tail (Kimi-K3: ``lm_head``, the drafter and sampling) starts with a spin."""


_NO_DELAYS = _Delays()
#: The wrap gathers (layers 1 and 2), the first and the last regular ones, and the one an abort in
#: the window of ``_MID`` leaves pending: a read that does not wait for its copy sees other bytes.
_COPY_DELAYS = _Delays(copies=frozenset({1, 2, 3, 4, _MID + 1, _LAST - 1, _LAST}))
#: The read windows at the step boundary and in the middle: a copy that does not wait for the
#: reads of its slot overwrites them.
_READ_DELAYS = _Delays(
    reads=frozenset({1, 2, _MID, _LAST - 1, _LAST}),
    aux=frozenset({1, 2, _LAST - 2, _LAST - 1, _LAST}),
    tail=True,
)

#: The graphs each runner captures, one-token decode batches, and the delays each bakes in.
_PLAIN = KeyType(1, 0)
_SLOW_COPIES = KeyType(1, 0, is_all_greedy_sample=False)
_SLOW_READS = KeyType(1, 0, short_seq_len_mode=True)
_GRAPHS = {_PLAIN: _NO_DELAYS, _SLOW_COPIES: _COPY_DELAYS, _SLOW_READS: _READ_DELAYS}

#: Exit status of the ``DEBUG=3`` child process once the device assert has ended it.
_DEVICE_ASSERT_EXIT = 3


class _Abort(Exception):
    """An error injected into a forward."""


def _linear(device: torch.device) -> nn.Linear:
    linear = nn.Linear(_HIDDEN, _HIDDEN, bias=False, dtype=torch.bfloat16, device=device)
    nn.init.normal_(linear.weight, std=_HIDDEN**-0.5)
    return linear


class _ToyLayer(nn.Module):
    """A decoder layer: ``attn`` is its attention segment, ``moe`` and ``shared`` its MoE one."""

    DENSE_DWDP_MULTI_STREAM_SAFE = True

    def __init__(self, layer_idx: int, device: torch.device) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.attn, self.moe, self.shared = _linear(device), _linear(device), _linear(device)

    def dense_dwdp_roots(self, segment: str) -> list[tuple[str, nn.Module]]:
        if self.layer_idx == 0:
            return []
        if segment == "attn":
            return [("attn", self.attn)]
        return [("moe", self.moe), ("shared", self.shared)]


class _ToyModel(nn.Module):
    """The causal LM: the decoder layers where setup finds them, an embedding and a head."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(_ToyLayer(idx, device) for idx in range(_NUM_LAYERS))
        self.embed = nn.Embedding(_VOCAB, _HIDDEN, dtype=torch.bfloat16, device=device)
        self.head = _linear(device)
        self.requires_grad_(False)


class _Executor:
    """``PyExecutor`` and ``model_engine`` around the toy model, with dense DWDP set up on it.

    A step runs on the execution stream, then the current stream waits for it and
    ``after_forward`` runs, as in ``PyExecutor._forward_step``.  A synchronizing CUDA call in a
    step raises, except in a capture and when the slot watchdog polls (``DEBUG >= 2``).  The
    faults: ``delays``; ``abort_at``, a covered layer whose read window raises, or ``"tail"``;
    ``fail_recovery``, which also fails the next copy launch, the recovery's first; and
    ``late_aux_join``, which joins the aux stream after H3 instead of before it.
    """

    def __init__(
        self,
        device: torch.device,
        manager: DenseDwdpManager,
        prime: PrimePolicy,
        debug: int,
        transfer: str,
    ) -> None:
        torch.manual_seed(0)
        self.device, self.manager, self.debug = device, manager, debug
        self.model = _ToyModel(device)
        self.reference = copy.deepcopy(self.model)
        self.execution_stream = torch.cuda.Stream(device)
        self.aux_stream = torch.cuda.Stream(device)
        self.delays = _NO_DELAYS
        self.abort_at: int | str | None = None
        self.fail_recovery = self.fail_launch = self.late_aux_join = False
        self.outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.runtime: DenseDWDP = manager.setup(
            self.model,
            comm=_LoopbackGroup(0, _NUM_RANKS),
            config=DenseConfig(enabled=True, strict=True, debug=debug, copy=transfer, prime=prime),
            device_id=device.index,
            store_factory=LoopbackShardStore,
            launcher_factory=self._launcher,
        )

    def _launcher(self, plans, transfer, device) -> Callable[[int], None]:
        """The production launcher of the transfer behind the copy delays and a launch failure."""
        launch = DenseCopyLauncher(plans, transfer.mode, device=device)

        def launcher(layer: int) -> None:
            if self.fail_launch:
                self.fail_launch = False
                raise RuntimeError(f"injected: the copy of layer {layer} failed")
            if layer in self.delays.copies:
                _spin(_SPIN)
            launch(layer)

        return launcher

    def forward(self, inputs: dict[str, torch.Tensor | None]) -> torch.Tensor:
        """``KimiLinearForCausalLM.forward``: the decoder and the tail in the forward scope."""
        with self.manager.forward_scope():
            self.manager.ensure_primed()  # H0 of KimiLinearModel.forward, a check under CLOSED
            return self._logits(self.model, inputs["input_ids"], self.runtime)

    def _logits(self, model: _ToyModel, ids: torch.Tensor, dense: DenseDWDP | None) -> torch.Tensor:
        x = model.embed(ids)
        for layer in model.model.layers:
            x = self._layer(layer, x, dense)
        if dense is not None:  # the tail, before the LM head
            if self.delays.tail:
                _spin(_SPIN)
            if self.abort_at == "tail":
                raise _Abort("in the tail")
        return torch.tanh(model.head(x))

    def _layer(self, layer: _ToyLayer, x: torch.Tensor, dense: DenseDWDP | None) -> torch.Tensor:
        """A decoder layer with the dense hooks at Kimi-K3's call sites."""
        idx = layer.layer_idx
        covered = dense is not None and dense.covers(idx)
        if covered:
            dense.wait_layer(idx)  # H1
            if self.abort_at == idx:
                self.fail_launch = self.fail_recovery
                raise _Abort(f"in the read window of layer {idx}")
            if idx in self.delays.reads:
                _spin(_SPIN)
        x = torch.tanh(layer.attn(x))
        if covered:
            dense.attention_done(idx)  # H2
        compute = torch.cuda.current_stream()
        aux = self.aux_stream if do_multi_stream() else compute
        aux.wait_stream(compute)
        with torch.cuda.stream(aux):
            if covered and idx in self.delays.aux:
                _spin(_SPIN)
            shared = layer.shared(x)
        moe = layer.moe(x)
        if not self.late_aux_join:
            compute.wait_stream(aux)
        if covered:
            dense.layer_done(idx)  # H3
        compute.wait_stream(aux)  # joined already, unless late_aux_join
        return torch.tanh(moe + shared)

    @contextlib.contextmanager
    def _step(self, synchronizes: bool = False) -> Iterator[None]:
        torch.cuda.set_sync_debug_mode("default" if synchronizes else "error")
        try:
            self.execution_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.execution_stream):
                yield
            torch.cuda.current_stream().wait_stream(self.execution_stream)
            torch.cuda.set_sync_debug_mode("default")  # the slot watchdog polls at DEBUG >= 2
            self.manager.after_forward()
        finally:
            torch.cuda.set_sync_debug_mode("default")

    def _inputs(self, tokens: int) -> dict[str, torch.Tensor | None]:
        """The forward's inputs as ``CUDAGraphRunner`` takes them, ``tokens`` random input ids."""
        input_ids = torch.randint(0, _VOCAB, (tokens,), dtype=torch.int32, device=self.device)
        positions = torch.zeros((1, tokens), dtype=torch.int32, device=self.device)
        return {"attn_metadata": None, "input_ids": input_ids, "position_ids": positions}

    def eager(
        self,
        tokens: int = 1,
        delays: _Delays = _NO_DELAYS,
        *,
        multi_stream: bool = False,
        abort_at: int | str | None = None,
        fail_recovery: bool = False,
    ) -> None:
        """An eager step (a warmup or a fallback); with ``abort_at`` its forward must raise."""
        inputs = self._inputs(tokens)
        self.delays, self.abort_at, self.fail_recovery = delays, abort_at, fail_recovery
        aborts = pytest.raises(_Abort) if abort_at is not None else contextlib.nullcontext()
        try:
            with aborts, self._step(), with_multi_stream(multi_stream):
                self.outputs.append((inputs["input_ids"], self.forward(inputs)))
        finally:
            self.delays, self.abort_at, self.fail_recovery = _NO_DELAYS, None, False

    def replay(self, runner: CUDAGraphRunner, key: KeyType) -> None:
        """A graph step: ``CUDAGraphRunner.replay``, behind the dense replay guard (its hook)."""
        inputs = self._inputs(1)
        with self._step():
            self.outputs.append((inputs["input_ids"], runner.replay(key, inputs).clone()))

    def capture(self) -> CUDAGraphRunner:
        """A new runner, warmed up and captured as ``model_engine`` does: a warmup-only pass over
        the graphs, then per graph a multi-stream warmup, the capture and a replay.  The replay
        guard is registered as the model registers it when the runner is created
        (``KimiLinearForCausalLM.register_cuda_graph_pre_replay_hooks``)."""
        runner = create_mock_cuda_graph_runner(1)
        assert self.manager.register_replay_guard(runner)
        with self._step(synchronizes=True), runner.allow_capture():
            for warmup_only in (True, False):
                runner.is_warmup_only = warmup_only
                for key, delays in _GRAPHS.items():
                    inputs = self._inputs(1)
                    self.delays = delays
                    runner.capture(key, self.forward, inputs)
                    self.delays = _NO_DELAYS
                    if not warmup_only:
                        output = runner.replay(key, inputs)
                        self.outputs.append((inputs["input_ids"], output.clone()))
            runner.is_warmup_only = False
        return runner

    def corrupted(self) -> list[tuple[int, float]]:
        """``(index, max|diff|)`` of the outputs since the last call that differ from the reference
        model's for their input ids (NaN differs too)."""
        torch.cuda.synchronize()
        diffs = [
            (output.float() - self._logits(self.reference, ids, None).float()).abs().max().item()
            for ids, output in self.outputs
        ]
        self.outputs.clear()
        return [(index, diff) for index, diff in enumerate(diffs) if diff != 0]


@contextlib.contextmanager
def _executor(
    device: torch.device, prime: PrimePolicy, debug: int = 1, transfer: str = "gather"
) -> Iterator[_Executor]:
    """An executor on a new toy model whose dense DWDP a graph-mode manager set up, strict, with
    ``transfer``; the manager stays registered meanwhile, as in a deployment."""
    mapping = _dep_mapping(_NUM_RANKS, 0)
    with DenseDwdpManager(DenseMode.STANDALONE, mapping=mapping, graph_mode=True) as manager:
        yield _Executor(device, manager, prime, debug, transfer)


def _transfer_env(patch: pytest.MonkeyPatch, transfer: str, device: torch.device) -> None:
    """The deployment's environment of ``transfer``: the gather's with the gather (which this
    build must have for the GPU), none of it with the copy engines."""
    for name, value in _GATHER_ENV.items():
        if transfer == "gather":
            patch.setenv(name, value)
        else:
            patch.delenv(name, raising=False)
    if transfer == "gather":
        _require_gather(device)


@pytest.fixture(scope="module")
def device() -> Iterator[torch.device]:
    """A GPU; skips otherwise."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    yield torch.device("cuda", torch.cuda.current_device())


@pytest.fixture(params=_TRANSFERS)
def transfer(request: pytest.FixtureRequest, device: torch.device) -> Iterator[str]:
    """``TRTLLM_DWDP_DENSE_COPY`` and its environment (``_transfer_env``)."""
    with pytest.MonkeyPatch.context() as patch:
        _transfer_env(patch, request.param, device)
        yield request.param


@pytest.mark.parametrize("debug", [1, 2], ids=["debug1", "debug2"])
@pytest.mark.parametrize("prime", list(PrimePolicy), ids=lambda prime: prime.value)
def test_graphs_and_eager_forwards_stay_bit_exact_in_any_order(device, transfer, prime, debug):
    """Random replays and eager forwards on two runners, and aborts followed directly by replays.

    The KV-estimation executor's runner captures the graphs and replays them among eager forwards
    (multi-stream execution on or off), then releases them; the final executor's runner does the
    same on the same runtime, then eager forwards abort in the read window of layer 1, ``_MID`` or
    ``_LAST`` or in the tail, each followed by replays.  Every output is bit-exact, with no demand
    fetch or fence fallback, one recovery per abort and, at ``DEBUG=2``, no watchdog mismatch.
    """
    rng = random.Random(f"{prime.value}-{debug}")
    with _executor(device, prime, debug, transfer) as executor:
        executor.eager()  # the first forward after setup, which primes under wrap
        runner = None
        for steps in (12, 40):  # the KV-estimation executor's graphs, then the final executor's
            if runner is not None:
                runner.clear()  # an executor releases its graphs when it shuts down
            runner = executor.capture()
            for _ in range(steps):
                if rng.random() < 0.6:
                    executor.replay(runner, rng.choice(list(_GRAPHS)))
                else:
                    delays = rng.choice(list(_GRAPHS.values()))
                    executor.eager(rng.randint(1, 4), delays, multi_stream=rng.random() < 0.3)
        for abort_at in (1, _MID, _LAST, "tail"):
            executor.eager(4, _COPY_DELAYS, abort_at=abort_at)
            for key in _GRAPHS:
                executor.replay(runner, key)
            executor.eager(3, _READ_DELAYS, multi_stream=True)
            executor.replay(runner, _PLAIN)
        assert executor.corrupted() == []
        runtime = executor.runtime
        assert (runtime.captures, runtime.recoveries) == (2 * len(_GRAPHS), 4)
        assert runtime.demand_fetches == runtime.war_fences == 0
        assert runtime.poll_watchdog() == [] and runtime.watchdog_mismatches == 0
        assert (runtime.watchdog_checks > 0) is (debug >= 2)


@pytest.mark.parametrize("removed", [False, True], ids=["kept", "removed"])
@pytest.mark.parametrize(
    "safeguard, prime, attribute",
    [
        ("H4 join", PrimePolicy.WRAP, "runtime._join"),
        ("recovery", PrimePolicy.WRAP, "runtime.recover"),
        ("entry barrier", PrimePolicy.ENTRY, "runtime._record_consume_events"),
        ("aux join before H3", PrimePolicy.WRAP, None),  # removed: the toy model joins after H3
        ("replay guard", PrimePolicy.WRAP, "manager.before_replay"),
        ("recovery join", PrimePolicy.ENTRY, "runtime._join"),
    ],
    ids=["h4_join", "recovery", "entry_barrier", "aux_join", "replay_guard", "recovery_join"],
)
def test_a_sequence_corrupts_an_output_only_without_its_safeguard(
    device, transfer, monkeypatch, safeguard, prime, attribute, removed
):
    """Negative controls: each sequence is bit-exact with its safeguard and corrupts without it.

    * H4 join: a replay right after an eager forward reads layers 1 and 2 before that forward's
      delayed wrap copies land.
    * recovery, and the replay guard after a failed recovery (``BROKEN``): a replay after an
      aborted eager forward reads the slots the dead forward refilled.
    * entry barrier: an eager forward's entry copies land before those of the replay before it,
      which then overwrites the slots of layers 1 and 2.
    * aux join before H3: the wrap copies at H3 overwrite the slots of the last two layers while
      their delayed aux reads still read them.
    * recovery join: with entry priming, the dead forward's delayed copy of ``_MID + 1`` lands
      after the copies of the replay that follows the abort.
    """
    with _executor(device, prime, transfer=transfer) as executor:
        executor.late_aux_join = removed and attribute is None
        executor.eager()
        runner = executor.capture()
        if removed and attribute is not None:
            owner, name = attribute.split(".")
            monkeypatch.setattr(getattr(executor, owner), name, lambda: None)
        if safeguard == "H4 join":
            for _ in range(4):
                executor.eager(1, _COPY_DELAYS)
                executor.replay(runner, _PLAIN)
        elif safeguard == "entry barrier":
            for _ in range(4):
                executor.replay(runner, _SLOW_READS)
                executor.eager()
        elif safeguard == "aux join before H3":
            executor.replay(runner, _SLOW_READS)
            executor.eager(1, _READ_DELAYS, multi_stream=True)
        else:  # an abort in the window of _MID, while its copy of _MID + 1 is delayed
            broken = safeguard == "replay guard"
            executor.eager(1, _COPY_DELAYS, abort_at=_MID, fail_recovery=broken)
            assert (executor.runtime.forward_state is ForwardState.BROKEN) is broken
            executor.replay(runner, _SLOW_READS)
            executor.replay(runner, _PLAIN)
        corrupted = executor.corrupted()
        assert bool(corrupted) is removed, corrupted


def _debug3_child(transfer: str) -> None:
    """The child process of the ``DEBUG=3`` test, in the environment of ``transfer``."""
    device = torch.device("cuda", torch.cuda.current_device())
    with pytest.MonkeyPatch.context() as patch:
        _transfer_env(patch, transfer, device)
        _debug3_forwards(device, transfer)


def _debug3_forwards(device: torch.device, transfer: str) -> None:
    with _executor(device, PrimePolicy.WRAP, debug=3, transfer=transfer) as executor:
        executor.eager()
        runner = executor.capture()
        for key in _GRAPHS:
            executor.replay(runner, key)
        executor.eager(2, _READ_DELAYS, multi_stream=True)
        assert executor.corrupted() == []
        # Layer 1's attention weights, in the slot the next replay reads before it gathers into it.
        executor.model.model.layers[1].attn.weight.fill_(1)
        try:
            executor.replay(runner, _PLAIN)
            torch.cuda.synchronize()
        except torch.AcceleratorError as error:
            if "device-side assert triggered" not in str(error):
                raise
            os._exit(_DEVICE_ASSERT_EXIT)  # the CUDA context is lost: no teardown


@pytest.mark.parametrize("copy", _TRANSFERS)
def test_debug3_stops_the_process_on_the_device_when_a_slot_holds_other_bytes(device, copy):
    """``DEBUG=3``: the device assert of H4 fires in a replay that read a slot with other bytes.

    In a child process, whose CUDA context the assert ends: steps under ``DEBUG=3`` are bit-exact
    and assert nothing; then layer 1's attention weights change in their slot, and the next replay
    fails on the device assert, before the host poll of the slot watchdog would.
    """
    here = Path(__file__).resolve().parent
    pythonpath = [str(here), str(here.parents[2]), os.environ.get("PYTHONPATH", "")]
    if copy == "gather":
        _require_gather(device)
    code = f"import {Path(__file__).stem} as test; test._debug3_child({copy!r})"
    child = subprocess.run(
        [sys.executable, "-c", code],
        cwd=here,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(pythonpath)},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert child.returncode == _DEVICE_ASSERT_EXIT, (child.stdout + child.stderr)[-3000:]


def test_the_capture_probe_passes_where_the_driver_captures_batched_copies(device):
    """The gate's probe: a captured ``cudaMemcpyBatchAsync`` replays and copies on this driver.

    Its replay is also what a graph-mode ``batched`` launch replays in every decode graph, so a
    driver that fails here fails the launch in the gate.
    """
    assert dense_plan.batched_copy_capture_error(device) is None


@pytest.mark.parametrize("failure", [RuntimeError, TypeError], ids=["cuda_error", "binding_error"])
def test_the_capture_probe_names_the_cuda_versions_when_the_capture_fails(
    device, monkeypatch, failure
):
    """A batched copy that fails in the capture: the probe reports it with the CUDA versions.

    Whatever the copy raises, a CUDA error or a binding that does not match (``TypeError``), the
    probe returns its verdict and ends its capture.
    """

    def failing_copy(*_args) -> None:
        raise failure("injected copy failure")

    monkeypatch.setattr(dense_plan, "_memcpy_batch", failing_copy)

    error = dense_plan.batched_copy_capture_error(device)

    assert error is not None
    assert error.startswith(
        "cudaMemcpyBatchAsync failed in a CUDA graph capture: injected copy failure (CUDA driver "
    ), error
    assert ", CUDA runtime " in error
    # The probe left no capture open, and the next probe with the real copy passes.
    assert not torch.cuda.is_current_stream_capturing()
    monkeypatch.undo()
    assert dense_plan.batched_copy_capture_error(device) is None
