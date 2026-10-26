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
"""CPU tests of the dense DWDP manager (``manager.DenseDwdpManager``).

* ``standalone_group`` (with a tag of its own).
* Graph mode, on a fake runtime that records the calls it gets: the protocol and priming setup
  commits, the forward scope (H0 and H4) and H0 inside it, executor priming, the replay guard and
  the per-step hook (watchdog polling, counter reports).
"""

from unittest.mock import MagicMock

import pytest
from torch import nn

from tensorrt_llm._torch.modules.dwdp.dense import manager as dense_manager
from tensorrt_llm._torch.modules.dwdp.dense.config import DenseConfig
from tensorrt_llm._torch.modules.dwdp.dense.manager import (
    STANDALONE_GROUP_TAG,
    DenseDwdpManager,
    StandaloneTransfer,
    set_global_dense_dwdp_manager,
    standalone_group,
)
from tensorrt_llm._torch.modules.dwdp.dense.types import (
    CopyMode,
    DenseMode,
    DenseProtocol,
    DenseProtocolError,
    ForwardState,
    PrimePolicy,
)
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _no_dense_manager():
    """No test starts or ends with a dense DWDP manager registered."""
    set_global_dense_dwdp_manager(None)
    yield
    set_global_dense_dwdp_manager(None)


def adp_mapping(size, rank=0):
    """Rank ``rank`` of an attention-DP instance of ``size`` ranks with experts parallel (DEP)."""
    return Mapping(
        world_size=size, rank=rank, tp_size=size, moe_ep_size=size, enable_attention_dp=True
    )


def calls(log):
    """``log`` without the event records."""
    return [entry for entry in log if entry != "record"]


# ---------------------------------------------------------------------------
# standalone_group and StandaloneTransfer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mapping, tp_group",
    [
        (adp_mapping(4, rank=2), [0, 1, 2, 3]),
        (
            Mapping(world_size=8, rank=5, tp_size=4, pp_size=2, enable_attention_dp=True),
            [4, 5, 6, 7],
        ),
    ],
    ids=["tp4", "second_pp_stage"],
)
def test_standalone_group_is_a_sub_communicator_over_the_tp_group(monkeypatch, mapping, tp_group):
    """The group is created with a tag of its own, not the default tag 0 of ``MPIDist.tp_comm``.

    ``MPIDist.tp_comm`` creates a group over the same ranks of the same parent, and MPI pairs the
    ``Create_group`` calls on one parent by their tag.
    """
    world = MagicMock()
    monkeypatch.setattr(dense_manager, "mpi_disabled", lambda: False)
    monkeypatch.setattr(dense_manager, "mpi_comm", lambda: world)

    group = standalone_group(mapping)

    assert mapping.tp_group == tp_group
    world.group.Incl.assert_called_once_with(tp_group)
    world.Create_group.assert_called_once_with(
        world.group.Incl.return_value, tag=STANDALONE_GROUP_TAG
    )
    assert group is world.Create_group.return_value
    # Not the default tag, and a valid tag under every MPI (MPI_TAG_UB >= 32767).
    assert 0 < STANDALONE_GROUP_TAG <= 32767


# ---------------------------------------------------------------------------
# Graph mode: the CLOSED protocol around a fake runtime
# ---------------------------------------------------------------------------


class FakeRuntime:
    """The ``DenseDWDP`` surface the manager drives; every call lands in ``log``."""

    layers = (1, 2, 3)

    def __init__(self):
        self.log = []
        self.forward_state = ForwardState.IDLE
        self.primed = self.forward_open = False
        self.captures = self.eager_forwards = self.recoveries = 0
        self.demand_fetches = self.war_fences = 0
        self.recovery_fails = False
        self.begin_error = None

    def begin_forward(self):
        self.log.append("begin")
        if self.begin_error is not None:
            raise self.begin_error
        self.forward_state = ForwardState.OPEN

    def end_forward(self, ok):
        self.log.append(("end", ok))
        if not ok and self.recovery_fails:
            raise RuntimeError("injected: the recovery failed too")
        self.forward_state = ForwardState.IDLE

    def recover(self):
        self.log.append("recover")
        self.forward_state = ForwardState.IDLE

    def poll_watchdog(self):
        self.log.append("poll")
        return []

    def require_eager(self, hook):
        self.log.append("require_eager")

    def prefetch(self, layer):
        self.log.append(("prefetch", layer))

    def mark_primed(self):
        self.log.append("mark_primed")
        self.primed = True

    def release(self):
        self.log.append("release")


class RootedLayer(nn.Module):
    def dense_dwdp_roots(self, segment):
        return []


@pytest.fixture
def committed(monkeypatch):
    """``make(graph_mode, **switches)``: a standalone manager whose setup committed a fake runtime.

    Returns the manager, the runtime and the keywords ``setup_dense_dwdp`` got.
    """

    def make(graph_mode=True, **switches):
        runtime, calls = FakeRuntime(), []

        def setup_dense_dwdp(model, transfer, comm, **kwargs):
            calls.append(kwargs)
            return runtime

        monkeypatch.setattr(dense_manager, "setup_dense_dwdp", setup_dense_dwdp)
        model = nn.Module()
        model.model = nn.Module()
        model.model.layers = nn.ModuleList([RootedLayer()])
        manager = DenseDwdpManager(
            DenseMode.STANDALONE, mapping=adp_mapping(4), graph_mode=graph_mode
        )
        manager.setup(
            model,
            comm=object(),
            transfer=StandaloneTransfer(CopyMode.BATCHED, lambda: None),
            config=DenseConfig(enabled=True, **switches),
            device_id=0,
        )
        return manager, runtime, calls[0]

    return make


@pytest.mark.parametrize(
    "graph_mode, prime, expected",
    [
        (True, None, (DenseProtocol.CLOSED, PrimePolicy.WRAP)),
        (True, "entry", (DenseProtocol.CLOSED, PrimePolicy.ENTRY)),
        (False, "wrap", (DenseProtocol.LEGACY, PrimePolicy.ENTRY)),
    ],
    ids=["graph", "graph-entry", "eager-ignores-prime"],
)
def test_setup_commits_the_protocol_of_the_mode(committed, graph_mode, prime, expected):
    manager, _, calls = committed(graph_mode=graph_mode, prime=prime)
    assert manager.graph_mode is graph_mode
    assert (calls["protocol"], calls["prime"]) == expected


def test_an_attached_manager_has_no_graph_mode():
    with pytest.raises(ValueError):
        DenseDwdpManager(DenseMode.ATTACHED, graph_mode=True)


@pytest.mark.parametrize("outcome", ["ok", "raises", "raises-and-recovery-fails", "refused"])
def test_forward_scope_ends_the_forward_with_its_outcome(committed, outcome):
    """H4 gets ``ok``; the forward's own exception propagates even when the recovery fails.

    A forward that H0 refused never began, so no H4 follows.  Either way the scope closes.
    """
    manager, runtime, _ = committed()
    runtime.recovery_fails = outcome == "raises-and-recovery-fails"
    error = RuntimeError("injected: the spec worker failed")
    if outcome == "refused":
        runtime.begin_error = error = DenseProtocolError("injected: H0 refused the forward")

    def forward():
        with manager.forward_scope():
            manager.ensure_primed()  # H0 inside the scope only checks it
            runtime.log.append("forward")
            if outcome != "ok":
                raise error

    if outcome == "ok":
        forward()
    else:
        with pytest.raises(RuntimeError) as caught:
            forward()
        assert caught.value is error
    runtime.begin_error = None
    with manager.forward_scope():  # not nested in the forward above
        pass
    first = ["begin"] if outcome == "refused" else ["begin", "forward", ("end", outcome == "ok")]
    assert runtime.log == [*first, "begin", ("end", True)]


def test_forward_scope_refuses_a_nested_forward(committed):
    manager, runtime, _ = committed()
    with pytest.raises(DenseProtocolError):
        with manager.forward_scope():
            with manager.forward_scope():
                pass
    # The refusal aborts the outer forward, which ends not ok; the next forward is fine.
    with manager.forward_scope():
        pass
    assert runtime.log == ["begin", ("end", False), "begin", ("end", True)]


@pytest.mark.parametrize("case", ["attached", "eager", "no-runtime"])
def test_forward_scope_does_nothing_outside_graph_mode(committed, case):
    if case == "attached":
        manager, runtime = DenseDwdpManager(DenseMode.ATTACHED), FakeRuntime()
        manager.attach(runtime, expert=None)
    elif case == "eager":
        manager, runtime, _ = committed(graph_mode=False)
    else:
        manager = DenseDwdpManager(DenseMode.STANDALONE, mapping=adp_mapping(4), graph_mode=True)
        runtime = FakeRuntime()
    with manager.forward_scope(), manager.forward_scope():
        pass
    assert runtime.log == []


@pytest.mark.parametrize("graph_mode", [True, False], ids=["graph", "eager"])
def test_priming_outside_a_forward_scope(committed, graph_mode):
    """Graph mode: executor-loop priming does nothing and H0 outside the scope is refused."""
    manager, runtime, _ = committed(graph_mode=graph_mode)
    manager.prefetch_first_layers()
    if graph_mode:
        with pytest.raises(DenseProtocolError):
            manager.ensure_primed()
        assert runtime.log == []
    else:
        manager.ensure_primed()  # LEGACY, as before: already primed by the executor loop
        assert runtime.log == [("prefetch", 1), ("prefetch", 2), "mark_primed", "require_eager"]


@pytest.mark.parametrize(
    "graph_mode, state",
    [(True, state) for state in ForwardState] + [(False, ForwardState.BROKEN)],
)
def test_before_replay_recovers_a_runtime_off_the_forward_boundary(
    committed, monkeypatch, graph_mode, state
):
    manager, runtime, _ = committed(graph_mode=graph_mode)
    logger = MagicMock()
    monkeypatch.setattr(dense_manager, "logger", logger)
    runtime.forward_state = state
    manager.before_replay()
    recovers = graph_mode and state is not ForwardState.IDLE
    assert runtime.log == (["recover"] if recovers else [])
    assert logger.error.call_count == int(recovers)


def test_after_forward_reports_counter_changes_sparsely_and_once_at_release(committed, monkeypatch):
    manager, runtime, _ = committed()
    logger = MagicMock()
    monkeypatch.setattr(dense_manager, "logger", logger)
    reported = []
    for step in range(1, 21):
        runtime.eager_forwards += step % 2  # the counters change at the 10 odd steps
        before = logger.info.call_count
        manager.after_forward()
        if logger.info.call_count > before:
            reported.append(step)
    assert reported == [1, 3, 7, 15]  # the 1st, 2nd, 4th and 8th change
    manager.cleanup()
    assert logger.info.call_count == len(reported) + 1
    assert runtime.log == ["release"]  # no watchdog polls without TRTLLM_DWDP_DENSE_DEBUG >= 2


@pytest.mark.parametrize(
    "graph_mode, debug, polls",
    [(True, 0, False), (True, 2, True), (True, 3, True), (False, 2, False)],
    ids=["debug0", "debug2", "debug3", "eager-debug2"],
)
def test_after_forward_polls_the_slot_watchdog_from_debug_2(committed, graph_mode, debug, polls):
    manager, runtime, _ = committed(graph_mode=graph_mode, debug=debug)
    manager.after_forward()
    assert runtime.log == (["poll"] if polls else [])
