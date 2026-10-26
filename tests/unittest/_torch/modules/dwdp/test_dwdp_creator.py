# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""PyExecutor construction tests for DWDP: manager ownership and the dense DWDP mode.

The dense DWDP gate's own rules and texts are tested in ``test_dense_gate.py``.
"""

import os
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm._torch.distributed import MPIDist
from tensorrt_llm._torch.modules.dwdp.dense.config import DENSE_ENV
from tensorrt_llm._torch.modules.dwdp.dense.manager import (
    get_global_dense_dwdp_manager,
    set_global_dense_dwdp_manager,
)
from tensorrt_llm._torch.modules.dwdp.dense.types import DenseMode
from tensorrt_llm._torch.pyexecutor import py_executor_creator
from tensorrt_llm._torch.pyexecutor.dwdp import (
    DwdpManager,
    get_global_dwdp_manager,
    set_global_dwdp_manager,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm.llmapi.llm_args import DwdpConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _no_global_managers():
    """Start and end every test without a registered DWDP or dense DWDP manager.

    A leftover registration would make the next ``DwdpManager`` or ``DenseDwdpManager`` of this
    process fail.
    """
    set_global_dwdp_manager(None)
    set_global_dense_dwdp_manager(None)
    yield
    set_global_dwdp_manager(None)
    set_global_dense_dwdp_manager(None)


def test_create_py_executor_rolls_back_new_dwdp_manager_on_error():
    manager = MagicMock()
    manager.__exit__.side_effect = lambda *_args: set_global_dwdp_manager(None)

    def fail_after_registration(**_kwargs):
        set_global_dwdp_manager(manager)
        raise RuntimeError("construction failed")

    with patch.object(
        py_executor_creator,
        "_create_py_executor_impl",
        side_effect=fail_after_registration,
    ):
        with pytest.raises(RuntimeError, match="construction failed"):
            py_executor_creator.create_py_executor(llm_args=MagicMock())

    manager.__exit__.assert_called_once_with(None, None, None)
    assert get_global_dwdp_manager() is None


def test_create_py_executor_leaves_a_pre_existing_dwdp_manager_alone():
    """The wrapper only tears down a manager it created, not one it inherited."""
    existing = MagicMock()
    set_global_dwdp_manager(existing)

    def fail_without_registering(**_kwargs):
        raise RuntimeError("construction failed")

    with patch.object(
        py_executor_creator,
        "_create_py_executor_impl",
        side_effect=fail_without_registering,
    ):
        with pytest.raises(RuntimeError, match="construction failed"):
            py_executor_creator.create_py_executor(llm_args=MagicMock())

    existing.__exit__.assert_not_called()
    assert get_global_dwdp_manager() is existing


# ----------------------------------------------------------------------
# Dense DWDP launch gate and mode
# ----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_dense_switches(monkeypatch):
    """Start every test without ``TRTLLM_DWDP_DENSE*`` variables; tests set the ones they need."""
    for name in [name for name in os.environ if name.startswith(DENSE_ENV)]:
        monkeypatch.delenv(name)


# ----------------------------------------------------------------------
# The final executor gets the DWDP managers
# ----------------------------------------------------------------------
#
# ``_create_py_executor_impl`` runs to its end with the harness of
# ``tests/unittest/_torch/executor/test_py_executor_creator_mla_cache_reuse_sync.py``
# (``_run_create_py_executor``): a stand-in model engine, sampler and ``KvCacheCreator``, CPU
# stand-ins for the CUDA calls, and a recording ``create_py_executor_instance``.  The executors it
# records are ``PyExecutor``s assembled without ``__init__``, as in ``test_dense_executor.py``, so
# a test can prime them and shut them down; the stand-in ``KvCacheCreator`` shuts the estimation
# executor down, as the real one does.  With expert DWDP the ``DwdpManager`` is the real one, its
# group and device work replaced as in ``test_dwdp_manager.py``.

#: The module of ``DwdpManager``, whose group and device work the expert DWDP runs replace.
_FAC = "tensorrt_llm._torch.pyexecutor.dwdp"


class _DummyCalibrator:
    def init(self, *args, **kwargs):
        return None

    def maybe_wrap_model(self, model):
        return model


class _DummyModel(SimpleNamespace):
    """A model ``DwdpManager.setup`` can keep a weak reference to, unlike a ``SimpleNamespace``."""


class _DummyModelEngine:
    """The model engine fields the creator reads, with a model it cannot run."""

    def __init__(self, *, attn_runtime_features, is_generation: bool = True) -> None:
        self.warmup_purpose = None
        self.attn_runtime_features = attn_runtime_features
        self.metrics = {}
        self.max_seq_len = 128
        self.max_num_tokens = 128
        self.max_num_seq_slots = 4
        self.model = _DummyModel(
            model_config=SimpleNamespace(
                enable_flash_mla=False,
                is_generation=is_generation,
                is_encoder_decoder=False,
                attn_backend="TRTLLM",
                sparse_attention_config=None,
                pretrained_config=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64),
                quant_config=QuantConfig(kv_cache_quant_algo=None),
            ),
            vocab_size_padded=32000,
        )

    def set_warmup_purpose(self, purpose: str) -> None:
        """Record the label the creator gives the following warmup passes."""
        self.warmup_purpose = purpose

    def release_executor_resources(self, *, release_model_owned_graphs: bool) -> None:
        """What ``PyExecutor`` shutdown calls on its model engine; there is nothing to release."""


class _ExpertWeights:
    """The expert weight manager ``setup_dwdp`` returns; logs each expert prefetch ``E``."""

    MOE_LAYERS = (3, 5, 7)

    def __init__(self, log: list[tuple]) -> None:
        self.log = log

    def first_moe_layer(self) -> int:
        return self.MOE_LAYERS[0]

    def next_moe_layer(self, layer: int) -> int | None:
        return next((later for later in self.MOE_LAYERS if later > layer), None)

    def prefetch_layer(self, layer: int) -> None:
        self.log.append(("E", layer))


class _DenseRuntime:
    """The runtime attached dense DWDP commits; logs each dense prefetch ``L`` and the release.

    It has no slot rings, so ``DwdpManager.setup`` reserves no storage for the expert setup.
    """

    def __init__(self, log: list[tuple]) -> None:
        self.log = log
        self.primed = False
        self.forward_open = False
        self.slots: dict = {}

    def prefetch(self, layer: int) -> None:
        self.log.append(("L", layer))

    def mark_primed(self) -> None:
        self.log.append(("mark_primed",))
        self.primed = True

    def release(self) -> None:
        self.log.append(("dense.release",))


def _final_executor_llm_args(
    dwdp_config: DwdpConfig | None = None, *, disable_overlap_scheduler: bool = True
) -> SimpleNamespace:
    """The ``llm_args`` fields ``_create_py_executor_impl`` reads to its end.

    Args:
        dwdp_config: The expert DWDP configuration; None runs without expert DWDP.
        disable_overlap_scheduler: Whether the overlap scheduler is off.
    """
    kv_cache_config = SimpleNamespace(
        enable_block_reuse=True,
        enable_partial_reuse=False,
        tokens_per_block=32,
        max_attention_window=None,
        mamba_state_config=SimpleNamespace(
            periodic_snapshot_interval=256,
            additional_snapshot_offsets_from_start=[],
            additional_snapshot_offsets_from_end=[],
        ),
    )
    return SimpleNamespace(
        garbage_collection_gen0_threshold=0,
        lora_config=None,
        kv_connector_config=None,
        scheduler_config=SimpleNamespace(
            context_chunking_policy=None, capacity_scheduler_policy=None
        ),
        peft_cache_config=SimpleNamespace(),
        kv_cache_config=kv_cache_config,
        decoding_config=None,
        guided_decoding_backend=None,
        custom_tokenizer=None,
        trust_remote_code=False,
        mm_encoder_only=False,
        enable_chunked_prefill=False,
        sparse_attention_config=None,
        attn_backend="TRTLLM",
        speculative_config=None,
        disable_overlap_scheduler=disable_overlap_scheduler,
        sleep_config=None,
        cache_transceiver_config=None,
        dwdp_config=dwdp_config,
        layer_wise_benchmarks_config=SimpleNamespace(
            calibration_mode=None,
            calibration_file_path=None,
            calibration_layer_indices=None,
        ),
        cuda_graph_config=None,
        parallel_config=SimpleNamespace(to_mapping=lambda: SimpleNamespace()),
        get_runtime_sizes=lambda: (1, 128, 128, 4),
    )


def _replace_dwdp_device_work(monkeypatch, log: list[tuple], *, dense_roots: bool) -> None:
    """Let the real ``DwdpManager`` run on CPU: replace its group, its setup and its teardown.

    The DWDP group is a mock that logs ``Free``, ``setup_dwdp`` returns an ``_ExpertWeights`` and
    ``teardown_dwdp`` is logged.  With ``dense_roots`` the model has dense DWDP roots and
    ``setup_dense_dwdp`` commits a ``_DenseRuntime``; without, there is no dense setup.
    """
    group = MagicMock()
    group.Free.side_effect = lambda: log.append(("group.Free",))
    comm_world = MagicMock()
    comm_world.Create_group.return_value = group
    comm_world.Get_size.return_value = 2
    monkeypatch.setattr(f"{_FAC}.COMM_WORLD", comm_world)
    monkeypatch.setattr(f"{_FAC}.dwdp_global_rank", lambda: 1)
    monkeypatch.setattr(py_executor_creator, "dwdp_global_rank", lambda: 1)
    monkeypatch.setattr(f"{_FAC}.torch.cuda.current_device", lambda: 0)
    monkeypatch.setattr(f"{_FAC}.setup_dwdp", lambda **_kwargs: _ExpertWeights(log))
    monkeypatch.setattr(
        f"{_FAC}.teardown_dwdp", lambda _model, expected_manager: log.append(("teardown_dwdp",))
    )
    monkeypatch.setattr(f"{_FAC}.dense_roots_available", lambda _model: dense_roots)
    monkeypatch.setattr(f"{_FAC}.setup_dense_dwdp", lambda *_args, **_kwargs: _DenseRuntime(log))


def _executor(
    log: list[tuple], kwargs: dict, start_worker_error: Exception | None = None
) -> PyExecutor:
    """The ``PyExecutor`` ``create_py_executor_instance(**kwargs)`` builds, without ``__init__``.

    It refuses and keeps the DWDP managers as ``PyExecutor.__init__`` does, and has what the
    creator, ``_prime_dwdp`` and ``shutdown`` read.  ``start_worker`` and ``shutdown`` are logged;
    ``start_worker`` then raises ``start_worker_error`` unless it is None.
    """
    dwdp_manager, dense_dwdp_manager = kwargs.get("dwdp_manager"), kwargs["dense_dwdp_manager"]
    PyExecutor._validate_dwdp_managers(
        dwdp_manager, dense_dwdp_manager, kwargs["llm_args"].disable_overlap_scheduler
    )
    executor = object.__new__(PyExecutor)
    executor._metrics = {}
    executor.peft_cache_config = kwargs["peft_cache_config"]

    def start_worker():
        log.append(("start_worker",))
        if start_worker_error is not None:
            raise start_worker_error

    executor.start_worker = start_worker
    executor.executor_request_queue = SimpleNamespace(
        enqueue_shutdown_request=lambda: log.append(("shutdown",))
    )
    executor.shutdown_event = SimpleNamespace(wait=lambda: None)
    executor._profile_manager = SimpleNamespace(cleanup=lambda: None)
    executor._openengine_node_server = None
    executor.hang_detector = SimpleNamespace(detected=lambda: False)
    executor.worker_thread = SimpleNamespace(join=lambda: None)
    executor.kv_connector_manager = None
    executor.dist = SimpleNamespace(pp_size=1)
    executor._shutdown_sleep_wakeup_listeners = lambda: None
    executor.model_engine = kwargs["model_engine"]
    executor.draft_model_engine = None
    executor.resource_manager = SimpleNamespace(resource_managers={})
    executor.virtual_memory_pools = None
    executor.sampler = SimpleNamespace()
    executor.dwdp_manager = dwdp_manager
    executor.dense_dwdp_manager = dense_dwdp_manager
    return executor


class _Creation(NamedTuple):
    """What ``_create_final_executor`` returns."""

    #: The call log; each ``create_py_executor_instance`` call logs ``("executor", kwargs)``.
    log: list[tuple]
    #: The executor ``create_py_executor`` returned.
    executor: PyExecutor
    #: The ``DwdpManager``; standalone, the stand-in ``DenseDwdpManager``.
    manager: object
    #: What ``_get_mapping`` returned.
    mapping: object


def _create_final_executor(
    monkeypatch,
    mode: str,
    *,
    estimating: bool,
    is_generation: bool = True,
    disable_overlap_scheduler: bool = True,
    start_worker_error: Exception | None = None,
    log: list[tuple] | None = None,
) -> _Creation:
    """Run ``create_py_executor`` to its end with DWDP in ``mode``.

    Args:
        mode: ``"standalone"``: dense DWDP without expert DWDP, on an attention-DP instance, with
            a stand-in ``DenseDwdpManager``.  ``"expert"``: expert DWDP alone, on a model without
            dense DWDP roots.  ``"attached"``: expert DWDP with dense DWDP attached.  With expert
            DWDP the ``DwdpManager`` is the real one (see ``_replace_dwdp_device_work``).
        estimating: What ``KvCacheCreator.try_prepare_estimation`` returns: whether a first
            executor only estimates the KV cache before the final one is built.
        is_generation: Whether the model is a generation model; only those have a KV cache.
        disable_overlap_scheduler: Whether the launch has the overlap scheduler off.
        start_worker_error: What the final executor's ``start_worker`` raises; None: it starts.
        log: The call log to fill; a new one if None.  A test whose creation raises passes its
            own, to read it afterwards.
    """
    log = [] if log is None else log
    managers = []

    class StandaloneManager:
        """Stands in for ``DenseDwdpManager``; ``setup`` would need a group and a GPU."""

        def __init__(self, mode, *, mapping=None):
            log.append(("DenseDwdpManager", mode, mapping))
            managers.append(self)

        def __enter__(self):
            log.append(("__enter__",))
            return self

        def setup(self, model):
            log.append(("setup", model))

    class LoggedDwdpManager(DwdpManager):
        """The real ``DwdpManager``, with its lifecycle calls logged."""

        def __init__(self, **kwargs):
            log.append(("DwdpManager",))
            super().__init__(**kwargs)
            managers.append(self)

        def __enter__(self):
            log.append(("DwdpManager.__enter__",))
            return super().__enter__()

        def setup(self, model):
            log.append(("DwdpManager.setup", model))
            return super().setup(model)

        def __exit__(self, *exc_info):
            log.append(("DwdpManager.__exit__",))
            return super().__exit__(*exc_info)

    class KvCacheCreator:
        def __init__(self, **kwargs):
            self._max_seq_len = kwargs["max_seq_len"]

        def try_prepare_estimation(self):
            return estimating

        def build_managers(self, resources, estimating_kv_cache):
            log.append(("build_managers", estimating_kv_cache))
            resources[ResourceManagerType.KV_CACHE_MANAGER] = SimpleNamespace()

        def configure_kv_cache_capacity(self, py_executor):
            log.append(("configure_kv_cache_capacity",))
            # The real one runs the estimation on the executor, then shuts it down.
            py_executor.shutdown_for_kv_cache_estimation()

        def teardown_managers(self, resources):
            log.append(("teardown_managers",))

    def create_py_executor_instance(**kwargs):
        log.append(("executor", kwargs))
        return _executor(log, kwargs, start_worker_error)

    dwdp_config = None
    if mode == "standalone":
        mapping = SimpleNamespace(
            rank=0,
            tp_size=4,
            pp_size=1,
            cp_size=1,
            has_pp=lambda: False,
            enable_attention_dp=True,
            is_last_pp_rank=lambda: True,
        )
    else:
        # A TP-1 context worker; the creator adds the DWDP fields.
        mapping = Mapping(world_size=1, rank=0)
        dwdp_config = DwdpConfig(
            dwdp_size=2, num_groups=1, num_experts_per_worker=4, num_prefetch_experts=4
        )
        _replace_dwdp_device_work(monkeypatch, log, dense_roots=mode == "attached")
    if mode != "expert":
        dense_mode = DenseMode.STANDALONE if mode == "standalone" else DenseMode.ATTACHED
        monkeypatch.setenv(DENSE_ENV, "1")
        monkeypatch.setattr(py_executor_creator, "validate_dense_gate", lambda *_args: dense_mode)
    monkeypatch.setattr(py_executor_creator, "DenseDwdpManager", StandaloneManager)
    monkeypatch.setattr(py_executor_creator, "DwdpManager", LoggedDwdpManager)
    monkeypatch.setattr(
        py_executor_creator,
        "_load_config_and_create_checkpoint_loader",
        lambda llm_args, checkpoint_dir: (llm_args, None),
    )
    monkeypatch.setattr(py_executor_creator, "_get_mapping", lambda _: mapping)
    monkeypatch.setattr(
        py_executor_creator.Distributed,
        "get",
        staticmethod(lambda mapping: MagicMock(spec=MPIDist)),
    )
    monkeypatch.setattr(
        py_executor_creator, "validate_feature_combination", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(py_executor_creator, "get_calibrator", lambda: _DummyCalibrator())
    monkeypatch.setattr(
        py_executor_creator, "instantiate_sampler", lambda *args, **kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(
        py_executor_creator, "get_spec_resource_manager", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(py_executor_creator, "get_spec_drafter", lambda *args, **kwargs: None)
    monkeypatch.setattr(py_executor_creator, "_adjust_torch_mem_fraction", lambda: None)
    monkeypatch.setattr(py_executor_creator, "log_memory_usage", lambda *args, **kwargs: None)
    monkeypatch.setattr(py_executor_creator, "is_mla", lambda _: True)
    monkeypatch.setattr(py_executor_creator, "is_hybrid_linear", lambda _: False)
    monkeypatch.setattr(py_executor_creator, "get_sm_version", lambda: 100)
    monkeypatch.setattr(py_executor_creator, "KvCacheCreator", KvCacheCreator)
    monkeypatch.setattr(py_executor_creator.torch.cuda, "mem_get_info", lambda: (2 << 30, 4 << 30))
    monkeypatch.setattr(py_executor_creator.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(py_executor_creator.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(
        py_executor_creator.torch.cuda, "memory_stats", lambda: {"allocated_bytes.all.current": 0}
    )
    monkeypatch.setattr(
        py_executor_creator.torch.cuda, "Stream", lambda: SimpleNamespace(cuda_stream=123)
    )
    # PyExecutor.shutdown synchronizes the device when CUDA is available.
    monkeypatch.setattr(py_executor_creator.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        py_executor_creator,
        "PyTorchModelEngine",
        lambda **kwargs: _DummyModelEngine(
            attn_runtime_features=kwargs["attn_runtime_features"], is_generation=is_generation
        ),
    )
    monkeypatch.setattr(
        py_executor_creator, "create_py_executor_instance", create_py_executor_instance
    )

    executor = py_executor_creator.create_py_executor(
        llm_args=_final_executor_llm_args(
            dwdp_config, disable_overlap_scheduler=disable_overlap_scheduler
        ),
        checkpoint_dir=None,
    )

    (manager,) = managers
    return _Creation(log, executor, manager, mapping)


def _calls(log: list[tuple]) -> list[tuple]:
    """``log`` with each executor's keyword arguments reduced to its two DWDP managers."""
    return [
        ("executor", entry[1].get("dwdp_manager"), entry[1]["dense_dwdp_manager"])
        if entry[0] == "executor"
        else entry
        for entry in log
    ]


def _logged_by(log: list[tuple], action) -> list[tuple]:
    """What ``action()`` appends to ``log``."""
    start = len(log)
    action()
    return log[start:]


def test_without_kv_cache_estimation_the_only_executor_gets_the_standalone_manager(monkeypatch):
    """The first executor is the final one: it primes the manager and exits it at shutdown.

    Dense DWDP is set up once, before ``build_managers`` sizes the final KV cache, so the KV
    cache gets the memory dense DWDP frees.
    """
    run = _create_final_executor(monkeypatch, "standalone", estimating=False)

    (executor,) = [entry[1] for entry in run.log if entry[0] == "executor"]
    model = executor["model_engine"].model
    assert _calls(run.log) == [
        ("DenseDwdpManager", DenseMode.STANDALONE, run.mapping),
        ("__enter__",),
        ("setup", model),
        ("build_managers", False),
        ("executor", None, run.manager),
        ("start_worker",),
    ]


def test_with_kv_cache_estimation_only_the_final_executor_gets_the_standalone_manager(monkeypatch):
    """The estimation executor gets None: its shutdown must not exit the manager.

    Dense DWDP is set up once, before the estimation run, which then measures the memory it
    frees.
    """
    run = _create_final_executor(monkeypatch, "standalone", estimating=True)

    executors = [entry[1] for entry in run.log if entry[0] == "executor"]
    model = executors[0]["model_engine"].model
    assert _calls(run.log) == [
        ("DenseDwdpManager", DenseMode.STANDALONE, run.mapping),
        ("__enter__",),
        ("build_managers", True),
        ("setup", model),
        ("executor", None, None),
        ("configure_kv_cache_capacity",),
        ("shutdown",),
        ("teardown_managers",),
        ("build_managers", False),
        ("executor", None, run.manager),
        ("start_worker",),
    ]


#: What ``PyExecutor._prime_dwdp`` enqueues through the ``DwdpManager`` before a forward: the
#: experts of the first two MoE layers, each after the layer's dense weights when dense DWDP is
#: attached.
_PRIMING = {
    "expert": [("E", 3), ("E", 5)],
    "attached": [("L", 3), ("E", 3), ("L", 5), ("E", 5), ("mark_primed",)],
}
#: What exiting the ``DwdpManager`` releases: dense DWDP first, then expert DWDP and its group.
_RELEASE = {
    "expert": [("teardown_dwdp",), ("group.Free",)],
    "attached": [("dense.release",), ("teardown_dwdp",), ("group.Free",)],
}


def _check_the_final_executor_drives_the_dwdp_manager(run: _Creation, mode: str) -> None:
    """It primes through the manager, and its shutdown exits the manager once.

    The exit releases DWDP and unregisters the manager and its dense manager.
    """
    assert run.executor.dwdp_manager is run.manager
    assert get_global_dwdp_manager() is run.manager
    assert _logged_by(run.log, run.executor._prime_dwdp) == _PRIMING[mode]
    assert _logged_by(run.log, run.executor.shutdown) == [
        ("shutdown",),
        ("DwdpManager.__exit__",),
        *_RELEASE[mode],
    ]
    assert run.executor.dwdp_manager is None
    assert get_global_dwdp_manager() is None
    assert get_global_dense_dwdp_manager() is None


@pytest.mark.parametrize("is_generation", [True, False], ids=["kv_cache", "no_kv_cache"])
@pytest.mark.parametrize("mode", ["expert", "attached"])
def test_without_kv_cache_estimation_the_only_executor_gets_the_dwdp_manager(
    monkeypatch, mode, is_generation
):
    """The first executor is the final one: it primes the ``DwdpManager`` and exits it at shutdown.

    Without the manager, nothing would prime the first two MoE layers' experts before a forward
    (with expert DWDP alone, those layers would read what the previous forward left in the
    slots), and nothing would release DWDP or unregister the managers.  With dense DWDP attached,
    the manager is set up before ``build_managers`` sizes the final KV cache, so the KV cache gets
    the memory dense DWDP frees; with expert DWDP alone, after it, as before.
    """
    run = _create_final_executor(monkeypatch, mode, estimating=False, is_generation=is_generation)

    (executor,) = [entry[1] for entry in run.log if entry[0] == "executor"]
    setup = [("DwdpManager.setup", executor["model_engine"].model)]
    kv_cache = [("build_managers", False)] if is_generation else []
    assert _calls(run.log) == [
        ("DwdpManager",),
        ("DwdpManager.__enter__",),
        *(setup + kv_cache if mode == "attached" else kv_cache + setup),
        ("executor", run.manager, None),
        ("start_worker",),
    ]
    _check_the_final_executor_drives_the_dwdp_manager(run, mode)


@pytest.mark.parametrize("mode", ["expert", "attached"])
def test_with_kv_cache_estimation_only_the_final_executor_gets_the_dwdp_manager(monkeypatch, mode):
    """Unchanged: the estimation executor gets no manager, so its shutdown exits none."""
    run = _create_final_executor(monkeypatch, mode, estimating=True)

    executors = [entry[1] for entry in run.log if entry[0] == "executor"]
    assert _calls(run.log) == [
        ("DwdpManager",),
        ("DwdpManager.__enter__",),
        ("build_managers", True),
        ("DwdpManager.setup", executors[0]["model_engine"].model),
        ("executor", None, None),
        ("configure_kv_cache_capacity",),
        ("shutdown",),
        ("teardown_managers",),
        ("build_managers", False),
        ("executor", run.manager, None),
        ("start_worker",),
    ]
    _check_the_final_executor_drives_the_dwdp_manager(run, mode)
