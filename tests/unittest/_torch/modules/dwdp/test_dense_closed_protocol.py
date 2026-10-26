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
"""CPU tests of the closed protocol of the dense DWDP runtime (``runtime.DenseDWDP``).

Under the closed protocol (standalone dense DWDP with CUDA graphs) every forward, eager, captured
or replayed, goes from the boundary state B back to B.  :class:`Model` drives the runtime as
``KimiLinearForCausalLM`` does inside ``DenseDwdpManager.forward_scope`` (H0, H4), with the
decoder's hooks H1-H3 and weight reads (one per read window on the auxiliary stream under
multi-stream execution), and as ``CUDAGraphRunner`` and the manager's replay guard do, on the
capture-aware fake of ``_capture_fake_ops``.  The vector-clock checker of ``test_dense_protocol``
then proves that every read sees its layer's copy and that no copy overwrites a slot before its
reads end, and :meth:`Model.assert_race_free` adds that no two copies into a slot overlap.
"""

from __future__ import annotations

import contextlib
import random
from collections.abc import Callable, Iterable

import pytest
from _capture_fake_ops import CaptureError, CaptureFakeOps, Graph, Stream
from test_dense_protocol import Entry, assert_race_free, happens_before, vector_clocks

from tensorrt_llm._torch.modules.dwdp.dense.runtime import DenseDWDP
from tensorrt_llm._torch.modules.dwdp.dense.types import (
    RING_DEPTH,
    ROW_ALIGN,
    ZERO_CHECKSUM,
    DenseProtocol,
    DenseProtocolError,
    ForwardState,
    PrimePolicy,
    SegmentKind,
    SlotSpec,
    allocate_slot_ring,
    closed_schedule,
)
from tensorrt_llm._torch.modules.dwdp.dense.watchdog import SlotWatchdog

pytestmark = pytest.mark.cpu_only

ATTN = SegmentKind.ATTN
MOE = SegmentKind.MOE
BOTH = (ATTN, MOE)
WRAP = PrimePolicy.WRAP
ENTRY = PrimePolicy.ENTRY
#: 8 % 2 == 92 % 2 and 8 % 3 == 92 % 3: the slots line up as for Kimi-K3's 92 covered layers.
LAYERS = tuple(range(1, 9))
#: ``(trigger layer, target layer)`` of the wrap gathers, per number of covered layers.
WRAPS = {
    1: [(1, 1)],
    2: [(1, 1), (2, 2)],
    3: [(3, 1), (3, 2)],
    8: [(7, 1), (8, 2)],
    92: [(91, 1), (92, 2)],
}

#: ``(hook, layer)``: where a forward is, ``("tail", None)`` after its last layer.
Point = tuple[str, int | None]


class Abort(Exception):
    """An error injected into a forward."""


class Model:
    """The causal LM around a committed runtime, and the executor that runs it.

    ``gathers`` lists ``(hook, trigger layer, target layer)`` of every copy launch, ``hook`` being
    the runtime call that launched it; a launch during ``failing_hook`` raises.
    """

    def __init__(
        self,
        layers: Iterable[int] = LAYERS,
        segments: tuple[SegmentKind, ...] = BOTH,
        prime: PrimePolicy = WRAP,
        *,
        protocol: DenseProtocol = DenseProtocol.CLOSED,
        multi_stream_ok: bool = True,
        strict: bool = False,
        watchdog: SlotWatchdog | None = None,
        pdl_break: bool = False,
    ) -> None:
        self.ops = CaptureFakeOps()
        self.layers = tuple(layers)
        self.segments = tuple(segments)
        slots = {
            kind: allocate_slot_ring(SlotSpec(kind, RING_DEPTH[kind], ROW_ALIGN, 2), "cpu")
            for kind in self.segments
        }
        self.runtime = DenseDWDP(
            self.layers,
            self.segments,
            slots,
            self.ops.copy,
            ops=self.ops,
            strict=strict,
            watchdog=watchdog,
            protocol=protocol,
            prime=prime,
            multi_stream_ok=multi_stream_ok,
            pdl_break=pdl_break,
        )
        self.runtime.set_launcher(self.launch)
        self.runtime.commit(None, fill_byte=0)
        self.graphs: dict[int, Graph] = {}
        self.hook: Point = ("commit", None)
        self.failing_hook: Point | None = None
        self.gathers: list[tuple[str, int | None, int]] = []

    def launch(self, layer: int) -> None:
        """The dense ``LayerLauncher``: one launch on the current stream."""
        if self.hook == self.failing_hook:
            raise RuntimeError(f"launch of L({layer}) failed")
        self.gathers.append((*self.hook, layer))
        self.ops.run("launch", self.ops.current, what="L", layer=layer)

    def read(self, kind: SegmentKind, layer: int) -> None:
        """A weight read; with multi-stream execution on the aux stream, forked and joined."""
        ops = self.ops
        if not ops.multi_stream:
            ops.run("read", ops.current, what=kind.value, layer=layer)
            return
        main = ops.current
        ops.aux.wait_stream(main)
        ops.run("read", ops.aux, what=kind.value, layer=layer)
        main.wait_stream(ops.aux)

    def forward(self, abort: Point | None = None) -> None:
        """One forward; ``abort``: raise :class:`Abort` right before that hook, or in the tail."""
        runtime = self.runtime

        def hook(name: str, layer: int, call: Callable[[int], None]) -> None:
            if abort == (name, layer):
                raise Abort(f"before {name}({layer})")
            self.hook = (name, layer)
            call(layer)

        self.hook = ("H0", None)
        runtime.begin_forward()
        ok = False
        try:
            for layer in self.layers:
                hook("H1", layer, runtime.wait_layer)
                self.read(ATTN, layer)
                hook("H2", layer, runtime.attention_done)
                if MOE in self.segments:
                    self.read(MOE, layer)
                hook("H3", layer, runtime.layer_done)
            if abort == ("tail", None):
                raise Abort("in the tail")
            ok = True
        finally:
            self.hook = ("H4", None)
            runtime.end_forward(ok)

    def capture(self, key: int, abort: Point | None = None) -> None:
        """``CUDAGraphRunner.capture``, then the replay of the new graph that follows it."""
        self.ops.multi_stream = True
        try:
            self.forward()  # the warmup forward
            with self.ops.capture() as graph:
                self.forward(abort)
        finally:
            self.ops.multi_stream = False
        self.graphs[key] = graph
        self.replay(key)

    def replay(self, key: int) -> None:
        """``CUDAGraphRunner.replay`` behind the replay guard of the dense manager."""
        if self.runtime.forward_state is not ForwardState.IDLE:
            self.runtime.recover()
        self.ops.replay(self.graphs[key])

    def assert_race_free(self) -> int:
        """``test_dense_protocol.assert_race_free``, then no two copies into a slot overlap.

        The read checks see a copy only from the latest copy of the read layer on.  A copy logged
        before that one but not ordered before it, such as one an aborted forward left on the copy
        stream while a replayed graph refills the slot, may land last unseen (WAW).
        """
        log = self.ops.log
        reads = assert_race_free(log, self.layers, self.segments)
        clocks = vector_clocks(log)
        position = {layer: index for index, layer in enumerate(self.layers)}
        last: dict[tuple[SegmentKind, int], int] = {}
        for index, entry in enumerate(log):
            if entry.kind != "launch":
                continue
            for kind in self.segments:
                slot = (kind, position[entry.layer] % RING_DEPTH[kind])
                previous = last.get(slot)
                assert previous is None or happens_before(log, clocks, previous, index), (
                    f"WAW: copy #{index} of L({entry.layer}) into {kind.value}[{slot[1]}] may "
                    f"overlap copy #{previous} of L({log[previous].layer})"
                )
                last[slot] = index
        return reads


@pytest.mark.parametrize("prime", [WRAP, ENTRY], ids=["wrap", "entry"])
@pytest.mark.parametrize("segments", [BOTH, (ATTN,)], ids=["attn+moe", "attn"])
@pytest.mark.parametrize("count", sorted(WRAPS))
def test_random_forwards_captures_replays_and_aborts_never_race(count, segments, prime):
    """Random interleavings on the execution stream, without host synchronization.

    Eager forwards with multi-stream execution on and off, captures and replays of two or three
    graphs, eager forwards and captures aborted before any hook or in the tail or by a copy launch
    that fails inside a hook, H0 included, and a second executor capturing its own graphs: every
    read sees its layer's copy, no copy overwrites a slot before its reads end or overlaps another
    copy into it, and the counters add up.
    """
    rng = random.Random(f"{count}-{len(segments)}-{prime.value}")
    model = Model(range(1, count + 1), segments, prime)
    runtime = model.runtime
    points = [(name, layer) for layer in model.layers for name in ("H1", "H2", "H3")]
    points.append(("tail", None))
    keys = rng.sample((1, 2, 4), rng.choice((2, 3)))
    expected = {"captures": 0, "recoveries": 0, "eager_forwards": 0}

    def new_executor() -> None:
        model.graphs.clear()
        model.forward()  # the first forward of an executor is eager
        for key in keys:
            model.capture(key)
        expected["eager_forwards"] += 1 + len(keys)
        expected["captures"] += len(keys)

    new_executor()
    # The hooks that launch copies in every forward, sorted so that the choice is reproducible.
    launches = sorted({(hook, layer) for hook, layer, _ in model.gathers if hook != "H0"})
    if prime is ENTRY:
        launches.append(("H0", None))
    actions = ("replay",) * 3 + ("eager",) * 2
    actions += ("abort", "abort capture", "failed launch", "new executor")
    for _ in range(24):
        action = rng.choice(actions)
        if action == "replay":
            model.replay(rng.choice(keys))
        elif action == "eager":
            model.ops.multi_stream = rng.random() < 0.5
            model.forward()
            model.ops.multi_stream = False
            expected["eager_forwards"] += 1
        elif action == "new executor":
            new_executor()
        elif action == "failed launch":
            capture = rng.random() < 0.5
            if capture:
                model.forward()  # the warmup forward before a capture
            model.failing_hook = rng.choice(launches)
            with pytest.raises(RuntimeError, match="launch of"):
                with model.ops.capture() if capture else contextlib.nullcontext():
                    model.forward()
            model.failing_hook = None
            model.replay(rng.choice(keys))  # the guard recovers a runtime left broken
            expected["eager_forwards"] += 1  # the failed forward, or the capture's warmup
            expected["recoveries"] += 1  # in place, or by the replay guard
        else:
            with pytest.raises(Abort):
                if action == "abort":
                    model.forward(abort=rng.choice(points))
                else:
                    model.capture(rng.choice(keys), abort=rng.choice(points))
            expected["eager_forwards"] += 1  # the aborted forward, or the capture's warmup
            expected["recoveries"] += 1
    model.replay(keys[0])  # the guard recovers from a capture that aborted last

    assert model.assert_race_free() > 0
    assert [entry for entry in model.ops.log if entry.kind == "sync"] == []
    assert {name: getattr(runtime, name) for name in expected} == expected
    assert runtime.demand_fetches == runtime.war_fences == 0


@pytest.mark.parametrize(
    "removed, prime",
    [
        ("entry barrier", ENTRY),
        ("join", WRAP),
        ("recovery", WRAP),
        ("recovery join", ENTRY),
        ("fresh event set", WRAP),
    ],
    ids=["no-barrier", "no-join", "no-recovery", "no-recovery-join", "no-fresh-events"],
)
def test_without_a_safeguard_a_replay_and_an_eager_forward_corrupt_a_slot(
    monkeypatch, removed, prime
):
    """Negative controls; the first four are left for the checker to find.

    Without the eager entry barrier, an eager forward's entry priming may overwrite slots that
    the replay before it still reads; without the eager join, a replay may read slots before the
    wrap copies of the eager forward before it land; without the recovery, a replay after an
    aborted forward reads slots the dead forward refilled; without the recovery's join, copies of
    the dead forward may land after the entry gathers of the replay that follows.  A capture that
    records the eager event set leaves it invalid, and the next eager forward fails on it.
    """
    model = Model(prime=prime)
    runtime = model.runtime
    model.forward()
    if removed == "fresh event set":
        events = iter((runtime._ready, runtime._consume))
        monkeypatch.setattr(runtime, "_new_ring_events", lambda: next(events))
    model.capture(1)
    if removed == "entry barrier":
        monkeypatch.setattr(runtime, "_record_consume_events", lambda: None)
        model.replay(1)
        model.forward()
    elif removed == "join":
        monkeypatch.setattr(runtime, "_join", lambda: None)
        model.forward()
        model.replay(1)
    elif removed in ("recovery", "recovery join"):
        if removed == "recovery":
            monkeypatch.setattr(runtime, "recover", lambda: None)
        else:  # with entry priming only the recovery joins between these forwards
            monkeypatch.setattr(runtime, "_join", lambda: None)
        with pytest.raises(Abort):
            model.forward(abort=("H2", 5))
        model.replay(1)
    else:
        with pytest.raises(CaptureError, match="InvalidValue"):
            model.forward()
        return
    with pytest.raises(AssertionError, match="RAW|WAR|WAW"):
        model.assert_race_free()


@pytest.mark.parametrize("prime", [WRAP, ENTRY], ids=["wrap", "entry"])
@pytest.mark.parametrize("segments", [BOTH, (ATTN,)], ids=["attn+moe", "attn"])
@pytest.mark.parametrize("count", sorted(WRAPS))
def test_a_forward_gathers_exactly_the_closed_schedule(count, segments, prime):
    """Each covered layer once per forward, at the hook ``closed_schedule`` names.

    ``L(l + 2)`` at H2(l); the first two layers at H0 (entry) or, for the next forward, at H3 of
    the layers whose slots they take over (wrap).
    """
    model = Model(range(1, count + 1), segments, prime)
    model.forward()  # with wrap priming the first forward after commit also primes at H0
    model.gathers.clear()
    model.forward()
    schedule = closed_schedule(model.layers, segments, prime)
    regular = [("H2", layer, layer + 2) for layer in model.layers[:-2]]
    if prime is ENTRY:
        expected = [("H0", None, layer) for layer in model.layers[:2]] + regular
    else:
        expected = regular + [("H3", trigger, target) for trigger, target in WRAPS[count]]
    assert [(e.hook, e.trigger_layer, e.target_layer) for e in schedule] == expected
    assert model.gathers == expected


def test_calls_out_of_protocol_are_refused_and_record_nothing():
    """The legacy protocol has no forward scope; the closed one primes and recovers itself.

    Under the closed protocol H1-H3 also need an open forward, and H4 with ``ok`` needs H3 of the
    last layer.
    """
    closed = Model()
    legacy = Model(protocol=DenseProtocol.LEGACY)
    sizes = (len(closed.ops.log), len(legacy.ops.log))
    calls = [
        lambda: closed.runtime.prefetch(1),
        closed.runtime.mark_primed,
        closed.runtime.war_barrier,
        lambda: closed.runtime.wait_layer(1),
        lambda: closed.runtime.end_forward(True),
        legacy.runtime.begin_forward,
        lambda: legacy.runtime.end_forward(True),
        legacy.runtime.recover,
    ]
    for call in calls:
        with pytest.raises(DenseProtocolError, match="protocol|forward state"):
            call()
    assert (len(closed.ops.log), len(legacy.ops.log)) == sizes


@pytest.mark.parametrize("prime", [WRAP, ENTRY], ids=["wrap", "entry"])
def test_a_capture_starts_only_in_the_boundary_state(prime):
    """B is ``IDLE`` and, with wrap priming only, primed; a capture outside it records nothing."""
    model = Model(prime=prime)
    runtime = model.runtime

    def capture_refused() -> bool:
        with model.ops.capture() as graph:
            try:
                runtime.begin_forward()
            except DenseProtocolError as error:
                assert "boundary state" in str(error) and graph.ops == []
                return True
            runtime.end_forward(False)
        return False

    assert capture_refused() is (prime is WRAP)  # after commit, nothing is primed
    model.forward()
    runtime.begin_forward()  # an eager forward is open
    assert capture_refused()
    runtime.end_forward(False)
    model.capture(1)
    assert runtime.captures == 1 and runtime.forward_state is ForwardState.IDLE


@pytest.mark.parametrize("capturing", [False, True], ids=["eager", "capture"])
@pytest.mark.parametrize("fault", ["demand fetch", "fence fallback"])
def test_a_capture_refuses_demand_fetches_and_fences_whatever_strict_says(fault, capturing):
    """Not strict, an eager forward falls back; a capture raises before it enqueues the fallback."""
    model = Model()
    runtime = model.runtime
    model.forward()
    with model.ops.capture() if capturing else contextlib.nullcontext():
        runtime.begin_forward()
        if fault == "fence fallback":  # H2(L2) refills D0 while L1's MoE-dense window is open
            runtime.wait_layer(1)
            runtime.wait_layer(2)
        gathers = len(model.gathers)
        refusal = pytest.raises(DenseProtocolError, match="capture forbids")
        with refusal if capturing else contextlib.nullcontext():
            if fault == "demand fetch":  # H1 of L1 and L2 did not run: L3's slots hold L1, L6
                runtime.wait_layer(3)
            else:
                runtime.attention_done(2)
        assert len(model.gathers) - gathers == int(not capturing)
        runtime.end_forward(False)
    expected = (0, 0) if capturing else ((1, 0) if fault == "demand fetch" else (0, 1))
    assert (runtime.demand_fetches, runtime.war_fences) == expected


@pytest.mark.parametrize("capturing", [False, True], ids=["eager", "capture"])
def test_multi_stream_execution_needs_the_declaration_of_the_layers(capturing):
    """Without ``multi_stream_ok``, H0 refuses multi-stream execution and records nothing."""
    model = Model(multi_stream_ok=False)
    runtime = model.runtime
    model.forward()
    model.ops.multi_stream = True
    entries = len(model.ops.log)
    with model.ops.capture() if capturing else contextlib.nullcontext() as graph:
        with pytest.raises(DenseProtocolError, match="multi-stream"):
            runtime.begin_forward()
    assert len(model.ops.log) == entries and not (graph and graph.ops)
    assert runtime.forward_state is ForwardState.IDLE


def test_a_failed_recovery_leaves_the_runtime_broken_until_the_next_one():
    """The forward's own error propagates; the replay guard then recovers before the replay."""
    model = Model()
    runtime = model.runtime
    model.forward()
    model.capture(1)
    model.failing_hook = ("H4", None)  # the recovery's priming fails
    with pytest.raises(Abort):
        model.forward(abort=("H2", 5))
    assert runtime.forward_state is ForwardState.BROKEN
    model.failing_hook = None
    model.replay(1)
    model.forward()
    assert runtime.recoveries == 2 and runtime.forward_state is ForwardState.IDLE
    model.assert_race_free()


@pytest.mark.parametrize("strict", [False, True], ids=["lenient", "strict"])
def test_the_watchdog_is_read_after_the_forward_and_asserted_at_h4(strict):
    """H3 reads no counters, nor does a poll in a capture; ``device_assert`` fails at H4.

    The poll after the forward returns the mismatches, or with ``strict`` raises them.
    """
    expected = {(layer, kind): (ROW_ALIGN, ZERO_CHECKSUM) for layer in LAYERS for kind in BOTH}
    watchdog = SlotWatchdog(LAYERS, BOTH, expected, "cpu", device_assert=True)
    model = Model(watchdog=watchdog, strict=strict)
    runtime = model.runtime
    model.forward()
    model.capture(1)
    assert runtime.poll_watchdog() == []  # the slots hold the zero blobs the checksums expect
    runtime.slots[ATTN][0].blob_view(1)[0] = 1  # A0 holds L1, L3, L5 and L7
    with pytest.raises(RuntimeError, match="slot watchdog"):
        model.forward()
    with model.ops.capture():
        assert runtime.poll_watchdog() == []
    if strict:
        with pytest.raises(DenseProtocolError, match="slot watchdog"):
            runtime.poll_watchdog()
    else:
        assert runtime.poll_watchdog() == [(layer, ATTN, 1) for layer in (1, 3, 5, 7)]


def assert_pdl_breaks(
    ops: list[Entry], start: int, end: int, origin: Stream, layers: tuple[int, ...]
) -> None:
    """The PDL breaks of the forward ``ops[start:end]``, whose execution stream is ``origin``.

    ``ops`` is a capture's operations or the log, with an eager forward or a replay at
    ``start:end``.  Two breaks per covered layer, on ``origin``, H1's and H2's in host order.
    H1's follows the latest copy of its layer, which the hook waited for, and precedes the
    layer's attention read; the copy behind that copy on the copy stream does not wait for it.
    H2's follows the attention read and precedes the MoE reads; the copy that H2 enqueues does
    not wait for it.
    """
    clocks = vector_clocks(ops)

    def before(first: int, second: int) -> bool:
        return happens_before(ops, clocks, first, second)

    def found(kind: str, what: str, layer: int) -> list[int]:
        return [
            index
            for index in range(start, end)
            if (ops[index].kind, ops[index].what, ops[index].layer) == (kind, what, layer)
        ]

    breaks = [index for index in range(start, end) if ops[index].kind == "break"]
    assert len(breaks) == 2 * len(layers)
    assert all(ops[index].stream is origin for index in breaks)
    for number, brk in enumerate(breaks):
        position, h2 = divmod(number, 2)
        layer = layers[position]
        (attention,) = found("read", ATTN.value, layer)
        if h2:
            assert before(attention, brk), f"H2({layer}): the attention read follows the break"
            assert all(before(brk, read) for read in found("read", MOE.value, layer)), layer
            if position + 2 < len(layers):
                (gather,) = found("launch", "L", layers[position + 2])
                assert not before(brk, gather), f"H2({layer}): its gather waits for the break"
            continue
        assert before(brk, attention), f"H1({layer}): the attention read precedes the break"
        copies = [i for i in range(brk) if (ops[i].kind, ops[i].layer) == ("launch", layer)]
        if not copies:  # a capture's entry barrier stands in for a copy made before the capture
            continue
        assert before(copies[-1], brk), f"H1({layer}): the break does not follow L({layer})"
        behind = next((i for i in range(copies[-1] + 1, end) if ops[i].kind == "launch"), None)
        assert behind is None or not before(brk, behind), f"H1({layer}): the next copy waits for it"


@pytest.mark.parametrize("prime", [WRAP, ENTRY], ids=["wrap", "entry"])
@pytest.mark.parametrize("pdl_break", [False, True], ids=["off", "on"])
def test_every_forward_launches_the_pdl_break_after_the_h1_waits_and_the_h2_gathers(
    pdl_break, prime
):
    """``pdl_break``: eager forwards (multi-stream execution off and on), a capture and its
    replays launch the PDL break at H1 and H2 of every covered layer (:func:`assert_pdl_breaks`),
    and the runtime creates it once, outside the captures; without it nothing is created or
    launched."""
    model = Model(prime=prime, pdl_break=pdl_break)
    ops = model.ops
    forwards = []

    def step(action: Callable[[], None], multi_stream: bool = False) -> None:
        start = len(ops.log)
        ops.multi_stream = multi_stream
        action()
        ops.multi_stream = False
        forwards.append((start, len(ops.log)))

    step(model.forward)
    model.capture(1)
    step(lambda: model.replay(1))
    step(model.forward, multi_stream=True)
    step(lambda: model.replay(1))
    graph = model.graphs[1]
    assert ops.pdl_breaks == int(pdl_break)
    if not pdl_break:
        assert [entry for entry in (*ops.log, *graph.ops) if entry.kind == "break"] == []
        return
    assert_pdl_breaks(graph.ops, 0, len(graph.ops), graph.origin, model.layers)
    for start, end in forwards:
        assert_pdl_breaks(ops.log, start, end, ops.exec, model.layers)
    assert model.assert_race_free() > 0
