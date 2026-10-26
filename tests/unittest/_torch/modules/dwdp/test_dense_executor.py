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
"""``PyExecutor`` and the DWDP managers: which one primes, which ones it accepts, the teardown.

With expert DWDP the executor holds the ``DwdpManager``, which drives the attached dense DWDP
manager; with standalone dense DWDP it holds the ``DenseDwdpManager`` itself.  Only the
non-overlap loop primes before a forward: in the overlap loop, which standalone dense DWDP
accepts, the model's forward entry (H0) primes every forward.  After every forward, in either
loop, the standalone manager's ``after_forward`` runs.

The executor is assembled with ``object.__new__`` and only the attributes a method reads, as the
executor tests do.
"""

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

pytestmark = pytest.mark.cpu_only

#: ``torch.cuda`` as ``py_executor`` reaches it.
_TORCH_CUDA = "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda"

#: ``PyExecutor``'s rejection of expert DWDP with the overlap scheduler, unchanged.
_OVERLAP_ERROR = "DWDP requires disable_overlap_scheduler=True. Overlap scheduler is not yet supported with DWDP."


def _manager(log: list[str], name: str) -> MagicMock:
    """A manager that logs ``<name>.prefetch_first_layers`` and ``<name>.__exit__``."""
    manager = MagicMock()
    manager.prefetch_first_layers.side_effect = lambda: log.append(f"{name}.prefetch_first_layers")
    manager.__exit__.side_effect = lambda *_args: log.append(f"{name}.__exit__")
    return manager


def test_prime_dwdp_primes_a_standalone_dense_manager():
    log = []
    executor = object.__new__(PyExecutor)
    executor.dwdp_manager = None
    executor.dense_dwdp_manager = _manager(log, "dense")

    executor._prime_dwdp()

    assert log == ["dense.prefetch_first_layers"]


@pytest.mark.parametrize("has_manager", [True, False], ids=["standalone", "no_manager"])
def test_after_forward_runs_once_per_forward_after_the_forward_is_joined(monkeypatch, has_manager):
    log = []
    current_stream = SimpleNamespace(wait_stream=lambda _stream: log.append("join"))
    monkeypatch.setattr(f"{_TORCH_CUDA}.current_stream", lambda: current_stream)
    monkeypatch.setattr(f"{_TORCH_CUDA}.stream", lambda _stream: contextlib.nullcontext())
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.py_executor.ExpertStatistic.set_iter",
        lambda _iteration: None,
    )
    # What ``PyExecutor._forward_step`` reads, as the executor tests of ``_forward_step`` build it.
    executor = SimpleNamespace(
        model_engine=SimpleNamespace(
            route_capture=None, forward=lambda *_args, **_kwargs: log.append("forward")
        ),
        resource_manager=None,
        iter_counter=0,
        enable_spec_decode=False,
        runtime_draft_len=0,
        sampler=SimpleNamespace(get_cache_indirection=lambda: None),
        execution_stream=SimpleNamespace(wait_stream=lambda _stream: None),
        _iter_adp_dummy_ctx_tokens=0,
        _iter_adp_dummy_gen_tokens=0,
        _compute_adp_dummy_tokens=PyExecutor._compute_adp_dummy_tokens,
        _maybe_record_hang_diagnostic_phase=lambda *_args, **_kwargs: None,
        _attach_encoder_output_to_execution_stream=lambda _requests: None,
        _mark_cross_kv_projection_consumed=lambda _requests: None,
        _kv_connector_wait_for_save=lambda: None,
        dense_dwdp_manager=(
            SimpleNamespace(after_forward=lambda: log.append("after_forward"))
            if has_manager
            else None
        ),
    )
    batch = SimpleNamespace(
        context_requests=[],
        generation_requests=[],
        num_context_requests=0,
        num_generation_requests=1,
    )

    for _ in range(2):
        PyExecutor._forward_step(executor, batch)

    assert log == ["forward", "join", *(["after_forward"] if has_manager else [])] * 2


# ---------------------------------------------------------------------------
# Accepted managers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("disable_overlap_scheduler", [True, False], ids=["no_overlap", "overlap"])
def test_a_standalone_dense_manager_runs_with_either_loop(disable_overlap_scheduler):
    PyExecutor._validate_dwdp_managers(None, MagicMock(), disable_overlap_scheduler)


def test_expert_dwdp_rejects_the_overlap_scheduler_as_before():
    with pytest.raises(ValueError) as error:
        PyExecutor._validate_dwdp_managers(MagicMock(), None, False)

    assert str(error.value) == _OVERLAP_ERROR


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


def _executor_to_shut_down(log: list, *, dwdp_manager, dense_dwdp_manager) -> PyExecutor:
    """An executor whose ``shutdown`` reaches the DWDP managers, assembled without ``__init__``."""
    executor = object.__new__(PyExecutor)
    executor.executor_request_queue = SimpleNamespace(enqueue_shutdown_request=lambda: None)
    executor.shutdown_event = SimpleNamespace(wait=lambda: None)
    executor._profile_manager = SimpleNamespace(cleanup=lambda: None)
    executor.hang_detector = SimpleNamespace(detected=lambda: False)
    executor.worker_thread = SimpleNamespace(join=lambda: log.append("worker joined"))
    executor.kv_connector_manager = None
    executor.dist = SimpleNamespace(pp_size=1)
    executor._shutdown_sleep_wakeup_listeners = lambda: None
    executor.model_engine = SimpleNamespace(release_executor_resources=lambda **_kwargs: None)
    executor.draft_model_engine = None
    executor.resource_manager = SimpleNamespace(
        resource_managers={"kv": SimpleNamespace(shutdown=lambda: log.append("kv.shutdown"))}
    )
    executor.virtual_memory_pools = None
    executor.sampler = SimpleNamespace()
    executor.dwdp_manager = dwdp_manager
    executor.dense_dwdp_manager = dense_dwdp_manager
    return executor


def _exiting(log: list, executor_box: list, name: str) -> MagicMock:
    """A manager whose ``__exit__`` logs whether the executor still holds its model engine."""
    manager = MagicMock()
    manager.__exit__.side_effect = lambda *_args: log.append(
        (f"{name}.__exit__", hasattr(executor_box[0], "model_engine"))
    )
    return manager


@pytest.fixture
def no_cuda(monkeypatch):
    """``shutdown`` synchronizes the device when CUDA is available; these executors have none."""
    monkeypatch.setattr(f"{_TORCH_CUDA}.is_available", lambda: False)


@pytest.mark.parametrize("mode", ["standalone", "attached"])
def test_shutdown_exits_the_manager_it_holds_after_the_model_is_dropped(no_cuda, mode):
    log, box = [], []
    manager = _exiting(log, box, mode)
    executor = _executor_to_shut_down(
        log,
        dwdp_manager=manager if mode == "attached" else None,
        dense_dwdp_manager=manager if mode == "standalone" else None,
    )
    box.append(executor)

    executor.shutdown()

    assert log == ["worker joined", "kv.shutdown", (f"{mode}.__exit__", False)]
    manager.__exit__.assert_called_once_with(None, None, None)
    assert executor.dwdp_manager is None and executor.dense_dwdp_manager is None
