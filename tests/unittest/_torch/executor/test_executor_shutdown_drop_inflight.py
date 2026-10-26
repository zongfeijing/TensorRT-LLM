# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A requested shutdown that drops in-flight requests instead of draining them.

``TRTLLM_EXECUTOR_SHUTDOWN_DROP_INFLIGHT=1`` lets the executor loop stop while
requests are still active or queued. Without it every case keeps the drain:
the loop stops only once nothing is left and no connector transfer is pending.
"""

import itertools
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from _torch.executor.kv_cache.test_kv_cache_v2_scheduler import (
    make_kv_cache_manager,
    make_scheduler,
)

from tensorrt_llm._torch.pyexecutor import py_executor as executor_module
from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor, _shutdown_drop_inflight_enabled
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import create_waiting_queue
from tensorrt_llm.logger import logger

pytestmark = pytest.mark.cpu_only

# The name deployments set. It is spelled out rather than imported, so renaming the
# executor's variable fails these tests instead of silently leaving the drain on.
_ENV = "TRTLLM_EXECUTOR_SHUTDOWN_DROP_INFLIGHT"

# Leaves `_drop_inflight_on_shutdown` unset, as on executors assembled without
# `__init__`.
_UNSET = object()

# Connector states: None is no connector; "idle" has nothing in flight; "load"
# has a prefix load in flight; "transfer" has a save in flight.
_CONNECTOR_STATES = (None, "idle", "load", "transfer")
_SHUTDOWN_CASES = list(itertools.product((False, True), (0, 2), (0, 3), _CONNECTOR_STATES))


class _KeptDraining(Exception):
    """Raised by the first step after the stop check: the loop kept going."""


class _HangDetector:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> bool:
        return False

    def checkpoint(self) -> None:
        pass


def _request(request_id: int) -> SimpleNamespace:
    """A request with the fields the termination path reads."""
    return SimpleNamespace(
        py_request_id=request_id,
        request_id=request_id,
        state=LlmRequestState.CONTEXT_INIT,
        is_dummy_request=False,
    )


def _executor(
    *, drop_inflight=_UNSET, active: int = 0, queued: int = 0, connector: str | None = None
) -> PyExecutor:
    """An executor asked to shut down, assembled without ``__init__``.

    Args:
        drop_inflight: The flag ``__init__`` resolves; ``_UNSET`` leaves it missing.
        active: Number of active requests.
        queued: Number of queued requests.
        connector: One of ``_CONNECTOR_STATES``.
    """
    executor = object.__new__(PyExecutor)
    if drop_inflight is not _UNSET:
        executor._drop_inflight_on_shutdown = drop_inflight
    executor.is_shutdown = True
    executor.active_requests = [_request(i) for i in range(active)]
    executor.waiting_queue = create_waiting_queue()
    executor.waiting_queue.add_requests(RequestQueueItem(100 + i) for i in range(queued))
    executor.kv_connector_manager = None
    if connector is not None:
        executor.kv_connector_manager = Mock()
        executor.kv_connector_manager.has_pending_loads.return_value = connector == "load"
        executor.kv_connector_manager.defer_load_termination.return_value = False
        executor.async_transfer_manager = Mock()
        executor.async_transfer_manager.has_any_inflight_requests.return_value = (
            connector == "transfer"
        )
        executor.async_transfer_manager.requests_in_transfer.return_value = {}
    # The termination path, down to the resource managers.
    executor.kv_cache_transceiver = None
    executor._disagg_pp_termination_handler = None
    executor._is_kv_manager_v2 = False
    executor.sampler = Mock()
    executor.resource_manager = Mock()
    executor._prefetched_request_ids = set()
    executor._disagg_coordinator = Mock()
    executor.model_engine = None
    executor.draft_model_engine = None
    executor.gather_all_responses = False
    executor.dist = SimpleNamespace(rank=0)
    executor.result_wait_queues = {}
    return executor


def _looping_executor(monkeypatch, **kwargs) -> PyExecutor:
    """An executor whose ``_executor_loop`` sees the shutdown on its first pass.

    A pass that goes on past the stop check raises ``_KeptDraining``.
    """
    monkeypatch.setattr("tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.set_device", Mock())
    monkeypatch.setattr("tensorrt_llm._torch.pyexecutor.py_executor.cudart.cudaSetDevice", Mock())
    monkeypatch.setattr("tensorrt_llm._torch.pyexecutor.py_executor.CUASSERT", Mock())
    executor = _executor(**kwargs)
    executor.is_shutdown = False

    def fetch_the_shutdown_request():
        executor.is_shutdown = True
        return []

    executor._fetch_and_activate_new_requests = Mock(side_effect=fetch_the_shutdown_request)
    executor._handle_control_request = Mock(side_effect=_KeptDraining)
    executor.device_id = 0
    executor._profiler = lambda: nullcontext(lambda: None)
    executor.hang_detector = _HangDetector()
    executor.enable_iter_perf_stats = False
    executor._resource_governor_enabled = False
    executor.is_benchmark_disagg = False
    executor._pending_transfer_responses = []
    executor._pending_response_terminations = []
    executor.enable_attention_dp = False
    executor._event_loop_completed = False
    return executor


@pytest.fixture
def log_calls(monkeypatch) -> Mock:
    """Capture what the TRT-LLM logger emits, with every log-once key unused."""
    calls = Mock()
    monkeypatch.setattr(logger, "log", calls)
    monkeypatch.setattr(logger, "_appeared_keys", set())
    return calls


def _messages(log_calls: Mock, severity: str) -> list[str]:
    return [
        " ".join(map(str, call.args[1:]))
        for call in log_calls.call_args_list
        if call.args[0] == severity
    ]


# ---- the gate __init__ resolves ----


def test_gate_is_off_and_silent_when_unset(monkeypatch, log_calls) -> None:
    monkeypatch.delenv(_ENV, raising=False)

    assert _shutdown_drop_inflight_enabled(1, True) is False
    log_calls.assert_not_called()


@pytest.mark.parametrize(
    ("raw", "pp_size", "disable_overlap_scheduler", "enabled", "logged"),
    [
        ("1", 1, True, True, "=1: dropping in-flight requests at shutdown is on"),
        ("1", 2, True, False, "=1: dropping in-flight requests at shutdown is off (needs pp_size"),
        ("1", 1, False, False, "=1: dropping in-flight requests at shutdown is off (needs pp_size"),
        ("1", 4, False, False, "=1: dropping in-flight requests at shutdown is off (needs pp_size"),
        ("0", 1, True, False, "='0': dropping in-flight requests at shutdown is off (only '1'"),
        ("true", 1, True, False, "='true': dropping in-flight requests at shutdown is off (only"),
        (" 1", 1, True, False, "=' 1': dropping in-flight requests at shutdown is off (only '1'"),
    ],
)
def test_gate_needs_one_pipeline_stage_without_the_overlap_scheduler(
    monkeypatch, log_calls, raw, pp_size, disable_overlap_scheduler, enabled, logged
) -> None:
    """A pipeline-parallel or overlap loop keeps draining whatever the variable says.

    A PP loop can stop with microbatches still travelling between ranks, and the
    overlap loop would skip processing its previous batch. The value that takes
    effect is logged once, so a run shows whether the variable reached the rank;
    "1" is logged bare and any other value quoted.
    """
    monkeypatch.setenv(_ENV, raw)

    assert _shutdown_drop_inflight_enabled(pp_size, disable_overlap_scheduler) is enabled

    (message,) = _messages(log_calls, logger.INFO)
    assert message.startswith(_ENV + logged), message


def _construct_executor(
    monkeypatch, *, pp_size: int, disable_overlap_scheduler: bool
) -> PyExecutor:
    """Run the real ``PyExecutor.__init__`` on the CPU, without starting its worker.

    The CUDA, request-queue and warmup collaborators are stand-ins, as in the
    constructor test of ``test_remote_tail_adp_phase.py``.
    """
    stream = SimpleNamespace(wait_stream=Mock())

    @contextmanager
    def stream_scope(_stream):
        yield

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "Stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "Event", Mock())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "stream", stream_scope)
    monkeypatch.setattr(executor_module, "_distributed_warmup_guard", lambda *_: nullcontext())
    monkeypatch.setattr(executor_module, "ExecutorRequestQueue", Mock())
    monkeypatch.setattr(executor_module, "mpi_disabled", lambda: True)
    monkeypatch.setattr(PyExecutor, "_set_global_steady_clock_offset", lambda _self: None)
    monkeypatch.setattr(PyExecutor, "_emit_initial_stats", lambda _self: None)
    monkeypatch.setenv("TRTLLM_GPU_KEEPALIVE", "0")
    monkeypatch.delenv("TLLM_TRACE_EXECUTOR_LOOP", raising=False)

    kv_cache_manager = SimpleNamespace(
        is_estimating_kv_cache=False,
        event_buffer_max_size=0,
        enable_block_reuse=False,
        tokens_per_block=128,
        get_kv_cache_stats=lambda: SimpleNamespace(max_num_blocks=0),
        get_max_resource_count=lambda: 0,
    )
    resource_manager = SimpleNamespace(
        resource_managers={ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager}
    )
    dist = SimpleNamespace(
        rank=0,
        tp_rank=0,
        tp_size=4,
        pp_size=pp_size,
        cp_size=1,
        mapping=object(),
        barrier=Mock(),
    )
    engine = SimpleNamespace(
        enable_attention_dp=True,
        spec_config=None,
        is_spec_decode=False,
        enable_in_graph_sampling=False,
        enable_overlap_headroom=False,
        mm_encoder_item_scheduling_enabled=False,
        cuda_graph_runner=SimpleNamespace(
            enabled=False, padding_enabled=True, padding_dummy_requests={}
        ),
        model=SimpleNamespace(
            model=SimpleNamespace(disagg_remote_tail_replay=False, disagg_context_only=True)
        ),
        llm_args=SimpleNamespace(
            max_stats_len=1,
            max_num_tokens=32,
            print_iter_log=False,
            enable_iter_perf_stats=False,
            enable_iter_req_stats=False,
            iter_perf_stats_interval=1,
            stream_interval=1,
            attention_dp_config=None,
            batch_wait_timeout_ms=0,
            batch_wait_timeout_iters=0,
            batch_wait_max_tokens_ratio=0,
        ),
        get_max_num_sequences=lambda: 8,
        warmup=lambda resources: None,
        warmup_scope=nullcontext,
    )
    return PyExecutor(
        resource_manager,
        make_scheduler(make_kv_cache_manager()),
        engine,
        SimpleNamespace(),
        dist,
        max_num_sequences=8,
        disable_overlap_scheduler=disable_overlap_scheduler,
        start_worker=False,
    )


@pytest.mark.parametrize(
    ("raw", "pp_size", "disable_overlap_scheduler", "enabled", "event_loop"),
    [
        ("1", 1, True, True, "_executor_loop"),
        ("1", 1, False, False, "_executor_loop_overlap"),
        ("1", 2, True, False, "_executor_loop_pp"),
        (None, 1, True, False, "_executor_loop"),
    ],
)
def test_init_resolves_the_flag_for_the_loop_it_runs(
    monkeypatch, raw, pp_size, disable_overlap_scheduler, enabled, event_loop
) -> None:
    """``__init__`` gates the flag on its own pipeline size and overlap setting.

    Only ``_executor_loop`` releases what dropped requests hold, so the flag is on
    only for the executor that runs that loop.
    """
    if raw is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, raw)

    executor = _construct_executor(
        monkeypatch, pp_size=pp_size, disable_overlap_scheduler=disable_overlap_scheduler
    )

    assert executor._drop_inflight_on_shutdown is enabled
    assert executor.event_loop == getattr(executor, event_loop)


# ---- should_stop_processing ----


@pytest.mark.parametrize("drop_inflight", [_UNSET, False])
@pytest.mark.parametrize(("is_shutdown", "active", "queued", "connector"), _SHUTDOWN_CASES)
def test_unset_keeps_the_drain_in_every_case(
    log_calls, drop_inflight, is_shutdown, active, queued, connector
) -> None:
    executor = _executor(
        drop_inflight=drop_inflight, active=active, queued=queued, connector=connector
    )
    executor.is_shutdown = is_shutdown

    drained = is_shutdown and active == 0 and queued == 0 and connector in (None, "idle")
    assert executor.should_stop_processing is drained
    assert _messages(log_calls, logger.WARNING) == []


@pytest.mark.parametrize(("is_shutdown", "active", "queued", "connector"), _SHUTDOWN_CASES)
def test_set_stops_a_requested_shutdown_unless_a_connector_transfer_is_pending(
    log_calls, is_shutdown, active, queued, connector
) -> None:
    """Work still active or queued no longer holds the loop; connector transfers still do.

    A connector load or save reads the KV pools that shutdown frees next, so it is
    waited for exactly as before.
    """
    executor = _executor(drop_inflight=True, active=active, queued=queued, connector=connector)
    executor.is_shutdown = is_shutdown

    assert executor.should_stop_processing is (is_shutdown and connector in (None, "idle"))


def test_dropping_is_warned_once_with_the_counts(log_calls) -> None:
    executor = _executor(drop_inflight=True, active=2, queued=3)

    assert executor.should_stop_processing
    assert executor.should_stop_processing

    (warning,) = _messages(log_calls, logger.WARNING)
    assert warning.startswith("shutdown requested: dropping 2 active and 3 queued requests")
    assert f"({_ENV}=1)" in warning


def test_an_idle_shutdown_drops_nothing_and_warns_nothing(log_calls) -> None:
    executor = _executor(drop_inflight=True)

    assert executor.should_stop_processing
    assert _messages(log_calls, logger.WARNING) == []


# ---- releasing what the dropped requests hold ----


def test_dropped_requests_release_through_the_termination_path() -> None:
    """Each dropped request hands its KV cache and slots back before shutdown frees the pools.

    A request whose KV cache the transfer fabric may still read keeps it, as on
    cancellation; the KV cache manager closes that cache at its own shutdown.
    """
    executor = _executor(drop_inflight=True, active=3, queued=2)
    released, in_transfer_state, pipelined = executor.active_requests
    in_transfer_state.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.has_inflight_transfer.side_effect = lambda req: req is pipelined
    executor.result_wait_queues = {req.py_request_id: Mock() for req in executor.active_requests}

    executor._release_requests_dropped_at_shutdown()

    executor.resource_manager.free_resources.assert_called_once_with(released)
    executor._disagg_coordinator.forget_request.assert_called_once_with(released.py_request_id)
    assert set(executor.result_wait_queues) == {
        in_transfer_state.py_request_id,
        pipelined.py_request_id,
    }
    assert executor.active_requests == []
    assert len(executor.waiting_queue) == 0


@pytest.mark.parametrize("drop_inflight", [_UNSET, False])
def test_unset_release_leaves_everything_in_place(drop_inflight) -> None:
    executor = _executor(drop_inflight=drop_inflight, active=2, queued=3)
    active = list(executor.active_requests)

    executor._release_requests_dropped_at_shutdown()

    assert executor.active_requests == active
    assert len(executor.waiting_queue) == 3
    executor.resource_manager.free_resources.assert_not_called()


# ---- the executor loop ----


@pytest.mark.parametrize("drop_inflight", [_UNSET, False])
def test_unset_loop_keeps_draining_with_work_pending(monkeypatch, drop_inflight) -> None:
    executor = _looping_executor(monkeypatch, drop_inflight=drop_inflight, active=2, queued=3)

    with pytest.raises(_KeptDraining):
        executor._executor_loop()

    assert not executor._event_loop_completed
    assert len(executor.active_requests) == 2
    executor.resource_manager.free_resources.assert_not_called()


@pytest.mark.parametrize("drop_inflight", [_UNSET, False, True])
def test_loop_stops_an_idle_executor(monkeypatch, drop_inflight) -> None:
    executor = _looping_executor(monkeypatch, drop_inflight=drop_inflight)

    executor._executor_loop()

    assert executor._event_loop_completed
    executor._handle_control_request.assert_not_called()


def test_set_loop_stops_with_work_pending_and_releases_after_its_last_collective(
    monkeypatch,
) -> None:
    """The release follows the loop's final response flush and its completion mark.

    The flush is the loop's last collective. Releasing after it keeps the
    release rank-local, and an error raised by it is reported as teardown
    rather than as a crash that strands peer ranks.
    """
    executor = _looping_executor(monkeypatch, drop_inflight=True, active=2, queued=3)
    order = []
    executor._flush_pending_transfer_responses = Mock(side_effect=lambda: order.append("flush"))
    executor.resource_manager.free_resources.side_effect = lambda req: order.append(
        ("free", req.py_request_id, executor._event_loop_completed)
    )

    executor._executor_loop()

    assert order == ["flush", ("free", 0, True), ("free", 1, True)]
    executor._handle_control_request.assert_not_called()
    assert executor.active_requests == []
    assert len(executor.waiting_queue) == 0


# ---- KV cache teardown ----


class _Cache:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Runtime:
    """The KV cache runtime's rule: its shutdown raises while any cache is open."""

    def __init__(self, caches: list[_Cache]) -> None:
        self._caches = caches
        self.shut_down = False

    def shutdown(self) -> None:
        still_open = [cache for cache in self._caches if not cache.closed]
        if still_open:
            raise RuntimeError(f"shutdown() with {len(still_open)} KV cache(s) still open")
        self.shut_down = True


@pytest.mark.parametrize("manager_type", [KVCacheManagerV2, MambaHybridCacheManagerV2])
def test_kv_teardown_closes_a_cache_a_dropped_request_kept(manager_type) -> None:
    """A cache no request freed is closed before the runtime's open-cache check.

    The release leaves a request the transfer fabric may still read holding its
    cache, as a context still in transfer does on any shutdown. If the manager's
    shutdown stopped closing such caches, the runtime would raise and skip the
    rest of the executor's teardown.
    """
    manager = object.__new__(manager_type)
    kept = _Cache()
    manager.kv_cache_map = {7: kept}
    manager.impl = _Runtime([kept])
    manager.sparse_metadata_batch = None
    manager.conversation_manager = None
    manager.event_manager = None
    manager._disagg_receive_ready = {}
    manager._on_allocation_stream = set()
    manager._allocation_ready = {}
    manager._request_stats_enabled_ids = set()
    manager._fresh_pages_filled = {}
    # Mamba hybrid state the subclass resets before delegating.
    manager._branch_snapshot_points = {}
    manager._kda_replay_suspended = set()
    manager._kda_replay_acceptance_pending = set()
    manager._kda_replay_unseeded = set()

    manager.shutdown()

    assert kept.closed
    assert manager.impl.shut_down
    assert manager.kv_cache_map == {}
