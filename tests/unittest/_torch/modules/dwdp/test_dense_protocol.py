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
"""CPU tests of the dense DWDP runtime protocol (``runtime.DenseDWDP``).

A recording fake of the runtime's CUDA seam logs every event record, stream wait and copy launch
in host order; a wait binds to the latest record of its event before it, as
``cudaStreamWaitEvent`` does.  A driver calls the runtime the way ``KimiLinearModel`` (H0-H3),
``ConfigurableMoE`` (the expert copies ``E(l)``, on the same copy stream) and ``DwdpManager``
(priming, WAR barrier) do, and logs the model's weight reads on the compute stream.  The tests
check the bindings the design lists (invariants I1-I4, table 2.4), that the host never
synchronizes after commit and, with vector clocks over the log, that no copy into a slot can
overlap a read of that slot.
"""

from __future__ import annotations

import contextlib
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import pytest

from tensorrt_llm._torch.modules.dwdp.dense.runtime import DenseDWDP
from tensorrt_llm._torch.modules.dwdp.dense.types import (
    EMPTY_TAG,
    RING_DEPTH,
    ROW_ALIGN,
    SegmentKind,
    SlotBuffer,
    SlotSpec,
    allocate_slot_ring,
)
from tensorrt_llm._torch.modules.dwdp.dense.watchdog import SlotWatchdog

pytestmark = pytest.mark.cpu_only

ATTN = SegmentKind.ATTN
MOE = SegmentKind.MOE
BOTH = (ATTN, MOE)
ATTN_ONLY = (ATTN,)
K3_LAYERS = tuple(range(1, 93))
#: Hook that closes each segment's read window (design 3.4.2).
WINDOW_END = {ATTN: "H2", MOE: "H3"}
#: ``(forward, hook, layer)`` of the hook during which a log entry was made.
Owner = tuple[int, str, int | None]
#: How ``DwdpManager.ensure_primed`` names itself when ``require_eager`` refuses it.
H0_HOOK = "DwdpManager.ensure_primed (H0)"

# ---------------------------------------------------------------------------
# Recording fake of the CUDA seam
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Entry:
    """One log entry, in host order.

    ``kind`` is ``record``, ``wait``, ``launch`` or ``read`` (stream operations), or ``mark``,
    ``nvtx``, ``sync``, ``detach``, ``store`` (host-side).  ``bound`` is the log index of the
    record a ``wait`` binds to (None: the event was never recorded).  ``what`` is ``L``/``E`` for
    a launch, the segment for a read, the hook for a mark and the range name for NVTX.
    """

    kind: str
    stream: FakeStream | None = None
    event: FakeEvent | None = None
    bound: int | None = None
    what: str = ""
    layer: int | None = None
    forward: int | None = None


class FakeStream:
    def __init__(self, log: list[Entry], name: str) -> None:
        self.log = log
        self.name = name

    def wait_event(self, event: FakeEvent) -> None:
        self.log.append(Entry("wait", stream=self, event=event, bound=event.last_record))

    def __repr__(self) -> str:
        return f"<stream {self.name}>"


class FakeEvent:
    def __init__(self, ops: FakeOps) -> None:
        self.ops = ops
        self.last_record: int | None = None

    def record(self, stream: FakeStream | None = None) -> None:
        stream = self.ops.current if stream is None else stream
        for observer in self.ops.record_observers:
            observer(stream)
        self.last_record = len(self.ops.log)
        self.ops.log.append(Entry("record", stream=stream, event=self))


class FakeOps:
    """``DeviceOps`` that logs every stream operation in host order."""

    def __init__(self) -> None:
        self.log: list[Entry] = []
        self.compute = FakeStream(self.log, "compute")
        self.copy = FakeStream(self.log, "copy")
        self.current = self.compute
        self.capturing = False
        self.multi_stream = False
        self.sync_error: Exception | None = None
        self.record_observers: list[Callable[[FakeStream], None]] = []

    def new_event(self) -> FakeEvent:
        return FakeEvent(self)

    def current_stream(self) -> FakeStream:
        return self.current

    @contextlib.contextmanager
    def stream(self, stream: FakeStream) -> Iterator[FakeStream]:
        previous, self.current = self.current, stream
        try:
            yield stream
        finally:
            self.current = previous

    def synchronize(self) -> None:
        self.log.append(Entry("sync"))
        if self.sync_error is not None:
            raise self.sync_error

    def is_current_stream_capturing(self) -> bool:
        return self.capturing

    def do_multi_stream(self) -> bool:
        return self.multi_stream

    @contextlib.contextmanager
    def nvtx_range(self, message: str) -> Iterator[None]:
        self.log.append(Entry("nvtx", what=message))
        yield


def make_ring(kind: SegmentKind, depth: int | None = None) -> tuple[SlotBuffer, ...]:
    depth = RING_DEPTH[kind] if depth is None else depth
    return allocate_slot_ring(SlotSpec(kind, depth, ROW_ALIGN, 2), "cpu")


def make_rings(segments: tuple[SegmentKind, ...] = BOTH) -> dict[SegmentKind, tuple]:
    return {kind: make_ring(kind) for kind in segments}


# ---------------------------------------------------------------------------
# Happens-before over the log
# ---------------------------------------------------------------------------


def vector_clocks(log: list[Entry]) -> dict[int, dict[FakeStream, int]]:
    """Vector clock of every stream operation: ``clock[s]`` = operations of ``s`` it follows.

    Streams run their operations in enqueue order; a wait makes the waiting stream follow every
    operation its bound record follows, the record included.
    """
    stream_clocks: dict[FakeStream, dict[FakeStream, int]] = {}
    clocks: dict[int, dict[FakeStream, int]] = {}
    for index, entry in enumerate(log):
        if entry.stream is None:
            continue
        clock = stream_clocks.setdefault(entry.stream, {})
        if entry.kind == "wait":
            assert entry.bound is not None, f"wait #{index} binds to an event never recorded"
            for stream, count in clocks[entry.bound].items():
                clock[stream] = max(clock.get(stream, 0), count)
        clock[entry.stream] = clock.get(entry.stream, 0) + 1
        clocks[index] = dict(clock)
    return clocks


def happens_before(
    log: list[Entry], clocks: dict[int, dict[FakeStream, int]], first: int, second: int
) -> bool:
    """Whether stream operation ``first`` completes before stream operation ``second`` starts."""
    stream = log[first].stream
    return first != second and clocks[second].get(stream, 0) >= clocks[first][stream]


# ---------------------------------------------------------------------------
# Driver: the model, ConfigurableMoE and DwdpManager around the runtime
# ---------------------------------------------------------------------------


class Driver:
    """Calls the runtime in the model's program order and logs reads and expert copies.

    Every runtime call is wrapped in a span keyed ``(forward, hook, layer)``; a ``mark`` entry
    opens it, so each log entry can be attributed to the hook that made it.
    """

    def __init__(
        self,
        *,
        segments: tuple[SegmentKind, ...] = BOTH,
        layers: tuple[int, ...] = K3_LAYERS,
        strict: bool = False,
        commit: bool = True,
        watchdog: SlotWatchdog | None = None,
    ) -> None:
        self.ops = FakeOps()
        self.log = self.ops.log
        self.segments = tuple(segments)
        self.layers = tuple(layers)
        self.slots = make_rings(self.segments)
        self.forward_index = 0
        self.fail_launch = False
        self.spans: dict[tuple[int, str, int | None], tuple[int, int]] = {}
        self.mark("construct", None)
        self.runtime = DenseDWDP(
            self.layers,
            self.segments,
            self.slots,
            self.ops.copy,
            ops=self.ops,
            strict=strict,
            watchdog=watchdog,
        )
        self.runtime.set_launcher(self.launch_dense)
        if commit:
            self.span("commit", None, lambda: self.runtime.commit(None, fill_byte=0))

    # -- logging -------------------------------------------------------------

    def mark(self, hook: str, layer: int | None) -> None:
        self.log.append(Entry("mark", what=hook, layer=layer, forward=self.forward_index))

    def span(self, hook: str, layer: int | None, action: Callable[[], object]) -> None:
        start = len(self.log)
        self.mark(hook, layer)
        action()
        self.spans[(self.forward_index, hook, layer)] = (start, len(self.log))

    def launch_dense(self, layer: int) -> None:
        """The dense ``LayerLauncher``: one launch on the current stream."""
        if self.fail_launch:
            self.fail_launch = False
            raise RuntimeError(f"launch of L({layer}) failed")
        self.log.append(Entry("launch", stream=self.ops.current, what="L", layer=layer))

    def launch_expert(self, layer: int) -> None:
        """``DWDPWeightManager.prefetch_layer``: ``E(layer)`` on the shared copy stream."""
        with self.ops.stream(self.ops.copy):
            self.log.append(Entry("launch", stream=self.ops.current, what="E", layer=layer))

    def read(self, kind: SegmentKind, layer: int) -> None:
        """A kernel that reads ``kind``'s weights of ``layer`` on the current stream."""
        self.log.append(Entry("read", stream=self.ops.current, what=kind.value, layer=layer))

    # -- DwdpManager -----------------------------------------------------------

    def prime_all(self) -> None:
        """``DwdpManager._prime_all``: ``L(first), E(first), L(second), E(second)``."""
        for layer in self.layers[:2]:
            self.runtime.prefetch(layer)
            self.launch_expert(layer)
        self.runtime.mark_primed()

    def ensure_primed(self) -> None:
        """``DwdpManager.ensure_primed`` at the model's H0 (design 3.4.6)."""
        self.runtime.require_eager(H0_HOOK)
        if self.runtime.forward_open:
            self.runtime.war_barrier()
        if not self.runtime.primed:
            self.prime_all()

    # -- model -----------------------------------------------------------------

    def layer(self, layer: int, *, stop_before: str | None = None) -> bool:
        """One decoder layer in program order; False if it stops before a hook."""
        runtime = self.runtime
        if stop_before == "H1":
            return False
        self.span("H1", layer, lambda: runtime.wait_layer(layer))
        self.read(ATTN, layer)  # attention(l): MKL:2053-2056
        if stop_before == "H2":
            return False
        self.span("H2", layer, lambda: runtime.attention_done(layer))
        if MOE in self.segments:
            self.read(MOE, layer)  # router and W-down
        self.span("experts", layer, lambda: None)  # CMOE:690, the first read of E(l)
        ahead = self.layers.index(layer) + 2
        if ahead < len(self.layers):
            self.span("C_E", layer, lambda: self.launch_expert(self.layers[ahead]))  # CMOE:709
        if MOE in self.segments:
            self.read(MOE, layer)  # W-up and the shared expert
        if stop_before == "H3":
            return False
        self.span("H3", layer, lambda: runtime.layer_done(layer))
        return True

    def forward(self, *, stop_at: tuple[str, int] | None = None, prime: bool = True) -> None:
        """One model forward; ``stop_at=(hook, layer)`` aborts it right before that hook."""
        if prime:
            self.span("H0", None, self.ensure_primed)
        for layer in self.layers:
            stop = stop_at[0] if stop_at is not None and stop_at[1] == layer else None
            if not self.layer(layer, stop_before=stop):
                break
        self.forward_index += 1

    # -- analysis --------------------------------------------------------------

    def owners(self) -> list[Owner | None]:
        """For every log index, the span (``(forward, hook, layer)``) that made the entry."""
        current = None
        result = []
        for entry in self.log:
            if entry.kind == "mark":
                current = (entry.forward, entry.what, entry.layer)
            result.append(current)
        return result

    def launch_index(self, what: str, layer: int, forward: int) -> int:
        owners = self.owners()
        found = [
            index
            for index, entry in enumerate(self.log)
            if entry.kind == "launch"
            and (entry.what, entry.layer) == (what, layer)
            and owners[index][0] == forward
        ]
        assert len(found) == 1, f"{what}({layer}) of forward {forward} launched at {found}"
        return found[0]

    def launches_in(self, key: tuple[int, str, int | None], what: str = "L") -> list[int]:
        """Log indices of the ``what`` launches made inside the span ``key``."""
        start, end = self.spans[key]
        return [
            index
            for index in range(start, end)
            if self.log[index].kind == "launch" and self.log[index].what == what
        ]

    def war_binding(self, launch: int) -> dict[str, list[tuple[int, int | None]]]:
        """``{hook: [(forward, layer)]}`` of the records that the copy waits before ``launch`` bind to.

        A consume wait binds to the latest consume record of its slot, a fence wait to the fence
        that the enqueueing hook records; both are compute-stream records.
        """
        owners = self.owners()
        result: dict[str, list[tuple[int, int | None]]] = {}
        for wait in self.copy_waits_before(launch):
            bound = self.log[wait].bound
            assert bound is not None and self.log[bound].stream is self.ops.compute
            forward, hook, layer = owners[bound]
            result.setdefault(hook, []).append((forward, layer))
        return result

    def copy_waits_before(self, launch: int) -> list[int]:
        """Copy-stream waits that the hook enqueueing ``launch`` made before it.

        They follow the hook's span mark and the copy-stream launch before ``launch``; waits that
        a failed launch left on the copy stream in an earlier span are not counted.
        """
        waits = []
        for index in range(launch - 1, -1, -1):
            entry = self.log[index]
            if entry.kind == "mark":
                break
            if entry.stream is not self.ops.copy:
                continue
            if entry.kind == "launch":
                break
            if entry.kind == "wait":
                waits.append(index)
        return waits[::-1]

    def last_copy_launch_before(self, index: int) -> Entry:
        for earlier in range(index - 1, -1, -1):
            entry = self.log[earlier]
            if entry.kind == "launch" and entry.stream is self.ops.copy:
                return entry
        raise AssertionError(f"no copy launch before #{index}")

    def expected_war_owner(self, kind: SegmentKind, layer: int, forward: int) -> Owner:
        """Whose consume record the copy of ``L(layer)`` into ``kind``'s ring waits for.

        The slot's previous holder: the layer one ring depth earlier in the same forward, else
        the last layer of the previous forward that used the slot, else the records of P6.
        """
        depth = RING_DEPTH[kind]
        position = self.layers.index(layer)
        if position >= depth:
            return (forward, WINDOW_END[kind], self.layers[position - depth])
        if forward == 0:
            return (0, "commit", None)
        previous = max(p for p in range(len(self.layers)) if p % depth == position % depth)
        return (forward - 1, WINDOW_END[kind], self.layers[previous])

    def assert_war_bindings(self) -> int:
        """Check that every dense copy waits for exactly the previous holders of its slots."""
        owners = self.owners()
        checked = 0
        for index, entry in enumerate(self.log):
            if entry.kind != "launch" or entry.what != "L":
                continue
            forward = owners[index][0]
            waits = self.copy_waits_before(index)
            assert all(self.log[self.log[w].bound].stream is self.ops.compute for w in waits)
            bound = Counter(owners[self.log[w].bound] for w in waits)
            expected = Counter(
                self.expected_war_owner(kind, entry.layer, forward) for kind in self.segments
            )
            assert bound == expected, f"L({entry.layer}) of forward {forward}"
            checked += 1
        return checked

    def assert_race_free(self) -> int:
        """Check every logged read against every copy into its slot; returns the reads checked.

        RAW: the latest copy of the read layer into the slot completes before the read.  WAR:
        every later copy into the slot starts after the read completes.
        """
        log = self.log
        clocks = vector_clocks(log)
        position = {layer: index for index, layer in enumerate(self.layers)}
        writes: dict[tuple[SegmentKind, int], list[tuple[int, int]]] = {}
        for index, entry in enumerate(log):
            if entry.kind == "launch" and entry.what == "L":
                for kind in self.segments:
                    key = (kind, position[entry.layer] % RING_DEPTH[kind])
                    writes.setdefault(key, []).append((index, entry.layer))
        reads = 0
        for index, entry in enumerate(log):
            if entry.kind != "read":
                continue
            kind = SegmentKind(entry.what)
            key = (kind, position[entry.layer] % RING_DEPTH[kind])
            slot_writes = writes.get(key, [])
            sources = [w for w, layer in slot_writes if w < index and layer == entry.layer]
            assert sources, f"read #{index} of {kind.value}({entry.layer}) precedes its copy"
            source = sources[-1]
            assert happens_before(log, clocks, source, index), (
                f"RAW: read #{index} of {kind.value}({entry.layer}) may start before copy #{source}"
            )
            for later, layer in slot_writes:
                if later > source:
                    assert happens_before(log, clocks, index, later), (
                        f"WAR: copy #{later} of L({layer}) into {kind.value}[{key[1]}] may start "
                        f"before read #{index} of {kind.value}({entry.layer}) completes"
                    )
            reads += 1
        return reads

    def assert_no_host_sync(self) -> None:
        """Check that the host never synchronized after commit, so it can run ahead of the GPU."""
        log = self.log
        committed = self.spans[(0, "commit", None)][1]
        syncs = [index for index in range(committed, len(log)) if log[index].kind == "sync"]
        assert syncs == [], f"host synchronize at log entries {syncs}"

    def compute_records(self, start: int, end: int) -> list[int]:
        return [
            index
            for index in range(start, end)
            if self.log[index].kind == "record" and self.log[index].stream is self.ops.compute
        ]

    def stream_ops(self, start: int, end: int | None = None) -> list[Entry]:
        return [e for e in self.log[start:end] if e.kind in ("record", "wait", "launch")]


def all_slots(segments: tuple[SegmentKind, ...] = BOTH) -> list[tuple[SegmentKind, int]]:
    return [(kind, index) for kind in segments for index in range(RING_DEPTH[kind])]


@pytest.fixture(scope="module", params=[BOTH, ATTN_ONLY], ids=["attn+moe", "attn"])
def two_forwards(request: pytest.FixtureRequest) -> Driver:
    """Two complete K3 forwards (92 layers each), primed at H0; tests only read it."""
    driver = Driver(segments=request.param)
    driver.forward()
    driver.forward()
    return driver


def test_i3_consume_events_are_recorded_only_where_read_windows_end(two_forwards):
    driver = two_forwards
    log = driver.log
    owners = driver.owners()
    committed = driver.spans[(0, "commit", None)][1]
    per_hook = Counter(owners[index][1] for index in driver.compute_records(committed, len(log)))
    expected = {"H2": 2 * len(driver.layers)}
    if MOE in driver.segments:
        expected["H3"] = 2 * len(driver.layers)
    assert per_hook == expected
    for forward in range(2):
        for layer in driver.layers:
            # H2 records first, so the copy of L(layer + 2) it enqueues binds to that record.
            start, end = driver.spans[(forward, "H2", layer)]
            ops = driver.stream_ops(start, end)
            assert ops[0].kind == "record" and ops[0].stream is driver.ops.compute
    # Every read of a window completes before any later copy into its slot starts.
    reads_per_layer = 3 if MOE in driver.segments else 1
    assert driver.assert_race_free() == 2 * len(driver.layers) * reads_per_layer


def test_last_layers_enqueue_nothing(two_forwards):
    driver = two_forwards
    log = driver.log
    for forward in range(2):
        for layer in driver.layers:
            start, end = driver.spans[(forward, "H2", layer)]
            launched = [
                (log[i].what, log[i].layer) for i in range(start, end) if log[i].kind == "launch"
            ]
            assert launched == ([("L", layer + 2)] if layer + 2 <= 92 else [])
            assert len(driver.compute_records(start, end)) == 1
    dense = Counter(entry.layer for entry in log if entry.kind == "launch" and entry.what == "L")
    assert dense == Counter({layer: 2 for layer in driver.layers})
    assert not driver.runtime.forward_open and not driver.runtime.primed


# ---------------------------------------------------------------------------
# Demand fetches and fences
# ---------------------------------------------------------------------------


def test_tag_mismatch_takes_one_fenced_demand_fetch():
    """A forward that skipped priming finds L(1) and L(2) missing and fetches each at H1."""
    driver = Driver()
    driver.forward(prime=False)
    runtime = driver.runtime
    assert runtime.demand_fetches == 2 and runtime.war_fences == 0
    log = driver.log
    for layer in (1, 2):
        start, end = driver.spans[(0, "H1", layer)]
        launches = [i for i in range(start, end) if log[i].kind == "launch"]
        assert [(log[i].what, log[i].layer) for i in launches] == [("L", layer)]
        # The copy waits for its slots' consume records (made at the end of setup) and for the
        # fence, recorded on compute at H1.
        assert driver.war_binding(launches[0]) == {"commit": [(0, None)] * 2, "H1": [(0, layer)]}
        # Compute then waits for the ready records that follow the demand copy.
        compute_waits = [
            i
            for i in range(launches[0], end)
            if log[i].kind == "wait" and log[i].stream is driver.ops.compute
        ]
        assert len(compute_waits) == 2 and all(log[i].bound > launches[0] for i in compute_waits)
        names = [entry.what for entry in log[start:end] if entry.kind == "nvtx"]
        assert names == [f"dwdp.dense.wait/L{layer}", f"dwdp.dense.prefetch/L{layer}"]
    for layer in driver.layers[2:]:
        start, end = driver.spans[(0, "H1", layer)]
        assert not [entry for entry in log[start:end] if entry.kind == "launch"]
    driver.forward()
    assert runtime.demand_fetches == 2
    driver.assert_race_free()
    driver.assert_no_host_sync()


@pytest.mark.parametrize("prime", [True, False], ids=["primed", "demand-fetched"])
def test_fence_when_a_holder_has_an_open_read_window(prime):
    """A forward dies inside attention(1); a prefetch that skips the WAR barrier gets the fence.

    H1(1) opened the read windows of ``A(1)`` and ``D(1)``, whether priming or H1's demand fetch
    enqueued ``L(1)``, and no consume record closed them: the consume events' latest records (from
    setup) precede the dead forward's reads.  The copy waits for those records and for the fence,
    recorded on compute after the reads.
    """
    driver = Driver()
    runtime = driver.runtime
    driver.forward(stop_at=("H2", 1), prime=prime)
    assert runtime.forward_open and runtime.demand_fetches == int(not prime)
    driver.span("again", None, lambda: runtime.prefetch(1))
    assert runtime.war_fences == 1  # one per prefetch, not one per slot
    (launch,) = driver.launches_in((1, "again", None))
    assert driver.war_binding(launch) == {"commit": [(0, None)] * 2, "again": [(1, None)]}
    driver.forward()  # H0 runs the barrier, then primes
    assert runtime.war_fences == 1 and runtime.demand_fetches == int(not prime)
    driver.assert_race_free()
    driver.assert_no_host_sync()


def test_war_barrier_then_reprime_after_an_abort():
    driver = Driver()
    runtime = driver.runtime
    driver.forward()
    driver.forward(stop_at=("H3", 41))  # dies inside the MoE block of layer 41
    assert runtime.forward_open and not runtime.primed
    start = len(driver.log)
    driver.span("barrier", None, runtime.war_barrier)
    records = driver.stream_ops(start)
    assert [entry.kind for entry in records] == ["record"] * (RING_DEPTH[ATTN] + RING_DEPTH[MOE])
    assert all(entry.stream is driver.ops.compute for entry in records)
    assert len({id(entry.event) for entry in records}) == len(records)
    for kind, index in all_slots():
        assert runtime.slot_tag(kind, index) == EMPTY_TAG and runtime.slot_consumed(kind, index)
    assert not runtime.primed and not runtime.forward_open
    driver.span("prime", None, driver.prime_all)
    owners = driver.owners()
    for layer in (1, 2):
        waits = driver.copy_waits_before(driver.launch_index("L", layer, driver.forward_index))
        assert Counter(owners[driver.log[w].bound][1] for w in waits) == Counter({"barrier": 2})
    assert runtime.primed and runtime.war_fences == 0
    driver.forward()
    assert runtime.demand_fetches == 0 and runtime.war_fences == 0
    driver.assert_race_free()
    driver.assert_no_host_sync()
