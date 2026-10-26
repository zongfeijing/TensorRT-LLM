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
"""CPU tests of the dense DWDP launch gate (``gate.validate_dense_gate``) in both modes.

The gate runs only when ``TRTLLM_DWDP_DENSE`` is on, before any manager or the model is built,
and returns the mode the launch runs dense DWDP in:

* attached, with expert DWDP: ``dwdp_config`` with ``dwdp_size > 1``, ``tp_size=1`` and the
  overlap scheduler off;
* standalone, on an attention-DP instance without expert DWDP (DEP16): no ``dwdp_config``,
  ``enable_attention_dp`` and ``tp_size > 1``; the overlap scheduler may be on.  With
  ``cuda_graph_config`` it runs in graph mode (``gate.dense_graph_mode``), as a generation server.

It checks, in order: the switches; the mode (a launch that fits neither is rejected with a
message naming both); ``TRTLLM_DWDP_DENSE_COPY`` (standalone: it must be valid; attached: any
value is ignored with one warning) and ``TRTLLM_DWDP_DENSE_PRIME`` (graph mode: it must be valid;
otherwise any value is ignored with one warning); the requirements both modes share, but for CUDA
graphs and speculation (attached and standalone without graphs: neither; graph mode: ``gather`` or
``batched`` set explicitly, and of the drafters only DSpark with a checkpoint of its own); and,
standalone only, the MPI backend, for ``gather`` the gather op, after which it warns once, and in
graph mode for ``batched`` a CUDA driver that captures ``cudaMemcpyBatchAsync``, after which it
logs the copy engines (INFO), or on Rubin (SM107) recommends the gather (one warning).

Every test starts without ``TRTLLM_DWDP_DENSE*`` variables, with MPI on, FABRIC peer handles, a
registered gather op, a driver that captures batched copies and an SM103 (GB300) GPU; a test
changes only what it checks.  The GB300 generation server shape (DEP16 with DSpark, CUDA graphs and
the copy engines) is accepted standalone in graph mode without an SM reserve; a Helix generation
server is rejected.
"""

import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest

from tensorrt_llm._torch.modules.dwdp import fabric_gather, vmm
from tensorrt_llm._torch.modules.dwdp.dense import gate
from tensorrt_llm._torch.modules.dwdp.dense.config import (
    DENSE_COPY_ENV,
    DENSE_ENV,
    DENSE_PDL_BREAK_ENV,
    DENSE_PRIME_ENV,
    DenseConfig,
)
from tensorrt_llm._torch.modules.dwdp.dense.gate import dense_graph_mode, validate_dense_gate
from tensorrt_llm._torch.modules.dwdp.dense.types import DenseMode
from tensorrt_llm._torch.peft.lora.config import LoraConfig
from tensorrt_llm._torch.pyexecutor import py_executor_creator
from tensorrt_llm.llmapi.llm_args import (
    CudaGraphConfig,
    DecodingBaseConfig,
    DraftTargetDecodingConfig,
    DSparkDecodingConfig,
    DwdpConfig,
    Eagle3DecodingConfig,
    KvCacheConfig,
    MoeConfig,
    MTPDecodingConfig,
    PrefillCudaGraphBackend,
    SleepConfig,
    TorchCompileConfig,
    TorchLlmArgs,
)
from tensorrt_llm.mapping import CpType, Mapping

pytestmark = pytest.mark.cpu_only

ATTACHED = DenseMode.ATTACHED
STANDALONE = DenseMode.STANDALONE
MODES = [ATTACHED, STANDALONE]

_HANDLE_TYPE = vmm.cuda.CUmemAllocationHandleType

#: Tensor parallelism of the accepted standalone launch: a Kimi-K3 DEP16 context instance.
STANDALONE_TP = 16

#: What a registered ``trtllm::dwdp_fabric_gather`` resolves to in these tests.
_GATHER_OP = object()
REJECTION_END = "Change the launch configuration or unset TRTLLM_DWDP_DENSE."

ATTACHED_COPY_WARNING = (
    "[DWDP Dense] TRTLLM_DWDP_DENSE_COPY={value!r} is ignored with expert DWDP: the dense copies "
    "run on the expert DWDP copy stream with the expert transfer"
)


#: The SM version of the tests' GPU: GB300's, whose generation servers use the copy engines.
_SM_VERSION = 103

#: The variables the dense gather's compute-side defaults set (``apply_compute_side_defaults``).
_GATHER_DEFAULTS = (
    "TRTLLM_CUTEDSL_SM_RESERVE",
    "TRTLLM_CUTEDSL_DENSE_GEMM_SCHEDULER",
    "TLLM_ROUTING_DISABLE_COOP_LAUNCH",
)

BATCHED_GRAPH_INFO = (
    "[DWDP Dense] TRTLLM_DWDP_DENSE_COPY=batched with CUDA graphs: every step copies every covered "
    "layer on the copy engines, which hold no SMs, so no SM reserve applies and the PDL break is off"
)
BATCHED_GRAPH_RUBIN_WARNING = (
    "[DWDP Dense] TRTLLM_DWDP_DENSE_COPY=batched with CUDA graphs on Rubin (SM107): every step "
    "copies every covered layer; generation servers were validated on this GPU with "
    "TRTLLM_DWDP_DENSE_COPY=gather, which is recommended here"
)


@pytest.fixture(autouse=True)
def _gate_host(monkeypatch) -> list[str]:
    """No ``TRTLLM_DWDP_DENSE*`` switch, MPI on, FABRIC peer handles, a registered gather op, a
    driver that captures batched copies, an SM103 GPU; returns the probe's calls."""
    for name in [name for name in os.environ if name.startswith(DENSE_ENV)]:
        monkeypatch.delenv(name)
    monkeypatch.setattr(gate, "mpi_disabled", lambda: False)
    monkeypatch.setattr(vmm, "peer_handle_type", lambda: _HANDLE_TYPE.CU_MEM_HANDLE_TYPE_FABRIC)
    monkeypatch.setattr(fabric_gather, "gather_op", lambda: _GATHER_OP)
    monkeypatch.setattr(gate, "get_sm_version", lambda: _SM_VERSION)
    probes = []
    monkeypatch.setattr(gate, "batched_copy_capture_error", lambda: probes.append("probe"))
    return probes


@pytest.fixture
def warnings(monkeypatch) -> list[str]:
    """The gate's own warnings, in order; neither its info lines nor ``config``'s warnings."""
    messages = []
    logger = SimpleNamespace(warning=messages.append, info=lambda message: None)
    monkeypatch.setattr(gate, "logger", logger)
    return messages


@pytest.fixture
def messages(monkeypatch) -> list[tuple[str, str]]:
    """The gate's own warnings and info lines, ``(level, message)`` in order."""
    logged = []
    logger = SimpleNamespace(
        warning=lambda message: logged.append(("warning", message)),
        info=lambda message: logged.append(("info", message)),
    )
    monkeypatch.setattr(gate, "logger", logger)
    return logged


def _launch(mode: DenseMode, **overrides) -> TorchLlmArgs:
    """The accepted context-worker launch of ``mode``, with ``overrides`` applied unvalidated."""
    dwdp_config = None
    if mode is ATTACHED:
        dwdp_config = DwdpConfig(
            dwdp_size=16, num_groups=1, num_experts_per_worker=56, num_prefetch_experts=56
        )
    args = TorchLlmArgs(
        model="/tmp/dummy_model",
        skip_tokenizer_init=True,
        cuda_graph_config=None,
        disable_overlap_scheduler=True,
        dwdp_config=dwdp_config,
    )
    return args.model_copy(update=overrides)


def _mapping(mode: DenseMode, **changes) -> Mapping:
    """The accepted mapping of ``mode`` (attached: tp 1; standalone: DEP16), with ``changes``."""
    fields = dict(tp_size=1, pp_size=1, cp_size=1, enable_attention_dp=False)
    if mode is STANDALONE:
        fields.update(tp_size=STANDALONE_TP, enable_attention_dp=True)
    fields.update(changes)
    if fields["enable_attention_dp"] and fields["cp_size"] == 1:
        # Experts parallel across the instance, as in DEP16; CP has no MoE EP.
        fields["moe_ep_size"] = fields["tp_size"]
    world_size = fields["tp_size"] * fields["pp_size"] * fields["cp_size"]
    return Mapping(world_size=world_size, rank=0, **fields)


class _Violation(NamedTuple):
    """A launch of one mode that violates exactly one requirement of the gate.

    With an empty ``message`` it describes a change of the accepted launch that the gate accepts.
    """

    name: str
    message: str
    """What the error must say: ``requires <requirement>, but <actual>`` or the mode's actual."""
    args: dict | None = None
    """``llm_args`` fields that differ from the mode's accepted launch."""
    mapping: dict | None = None
    """``Mapping`` arguments that differ from the mode's accepted mapping."""
    env: dict | None = None
    """``TRTLLM_DWDP_DENSE*`` switches to set."""
    lacks: frozenset = frozenset()
    """What the host lacks: ``fabric`` peer handles, ``mpi``, the ``gather_op``."""


#: The attached requirements other than the DWDP group size, which has a test of its own.
_ATTACHED_MODE = [
    _Violation("tp_size", "this launch has dwdp_config with tp_size=2", mapping={"tp_size": 2}),
    _Violation(
        "overlap_scheduler",
        "this launch has dwdp_config with disable_overlap_scheduler=False",
        args={"disable_overlap_scheduler": False},
    ),
]

#: The requirements of both modes, in the order the gate checks them; CUDA graphs and
#: speculation differ by mode (``test_graph_mode_and_drafters``).
_COMMON = [
    _Violation(
        "torch_compile_config",
        "requires torch_compile_config=None, but torch_compile_config is set",
        args={"torch_compile_config": TorchCompileConfig()},
    ),
    _Violation(
        "prefill_cuda_graph_backend",
        "requires prefill_cuda_graph_backend='disabled', but prefill_cuda_graph_backend=",
        args={"prefill_cuda_graph_backend": PrefillCudaGraphBackend.BREAKABLE},
    ),
    _Violation(
        "sleep_config",
        "requires sleep_config=None, but sleep_config is set",
        args={"sleep_config": SleepConfig()},
    ),
    _Violation(
        "lora_config",
        "requires lora_config=None, but lora_config is set",
        args={"lora_config": LoraConfig()},
    ),
    _Violation(
        "enable_locality_domains",
        "requires enable_locality_domains=False, but enable_locality_domains=True",
        args={"enable_locality_domains": True},
    ),
    _Violation("pp_size", "requires pp_size=1, but pp_size=2", mapping={"pp_size": 2}),
    _Violation("cp_size", "requires cp_size=1, but cp_size=2", mapping={"cp_size": 2}),
    _Violation(
        "peer_handle_type",
        "requires FABRIC peer handles (CU_MEM_HANDLE_TYPE_FABRIC), but this host shares "
        "CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR handles",
        lacks=frozenset({"fabric"}),
    ),
]

_CASES = [(ATTACHED, violation) for violation in _ATTACHED_MODE] + [
    (mode, violation) for mode in MODES for violation in _COMMON
]
_CASE_IDS = [f"{mode.value}-{violation.name}" for mode, violation in _CASES]


def _check(monkeypatch, mode: DenseMode, violation: _Violation | None = None) -> DenseMode:
    """Run the gate on the accepted launch of ``mode`` changed by ``violation``."""
    violation = violation or _Violation("none", "")
    for name, value in (violation.env or {}).items():
        monkeypatch.setenv(name, value)
    if "fabric" in violation.lacks:
        handle_type = _HANDLE_TYPE.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        monkeypatch.setattr(vmm, "peer_handle_type", lambda: handle_type)
    if "mpi" in violation.lacks:
        monkeypatch.setattr(gate, "mpi_disabled", lambda: True)
    if "gather_op" in violation.lacks:
        monkeypatch.setattr(fabric_gather, "gather_op", lambda: None)
    llm_args = _launch(mode, **(violation.args or {}))
    return validate_dense_gate(llm_args, _mapping(mode, **(violation.mapping or {})))


# ---------------------------------------------------------------------------
# Accepted launches and the mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_dense_gate_accepts_a_dense_dwdp_context_worker(monkeypatch, warnings, mode):
    assert _check(monkeypatch, mode) is mode
    assert warnings == []


@pytest.mark.parametrize(
    "dwdp_config, mapping, actual",
    [
        (DwdpConfig(dwdp_size=1), {}, "dwdp_config with dwdp_size=1"),
        (None, {}, "no dwdp_config and enable_attention_dp=False"),
        (None, {"enable_attention_dp": True}, "no dwdp_config and tp_size=1"),
    ],
    ids=["dwdp_size_1", "no_dwdp_config", "attention_dp_tp1"],
)
def test_dense_gate_requires_a_dwdp_group(monkeypatch, dwdp_config, mapping, actual):
    """Without a DWDP group of two or more ranks, only an attention-DP group of them will do."""
    violation = _Violation("group", "", args={"dwdp_config": dwdp_config}, mapping=mapping)
    with pytest.raises(ValueError, match=re.escape(f"but this launch has {actual}.")):
        _check(monkeypatch, ATTACHED, violation)


@pytest.mark.parametrize("mode, violation", _CASES, ids=_CASE_IDS)
def test_dense_gate_names_the_violated_requirement(monkeypatch, mode, violation):
    with pytest.raises(ValueError, match=re.escape(violation.message)) as error:
        _check(monkeypatch, mode, violation)
    message = str(error.value)
    assert message.startswith(f"Dense DWDP ({DENSE_ENV}) ")
    assert message.endswith(REJECTION_END)


@pytest.mark.parametrize("value", ["bogus", "batch", "per-slice", "copy_engine"])
def test_standalone_rejects_an_invalid_copy_switch(monkeypatch, warnings, value):
    with pytest.raises(ValueError) as error:
        _check(monkeypatch, STANDALONE, _Violation("copy", "", env={DENSE_COPY_ENV: value}))
    assert str(error.value) == (
        f"Dense DWDP ({DENSE_ENV}) requires valid TRTLLM_DWDP_DENSE_* switches, but "
        f"{DENSE_COPY_ENV}={value!r} must be one of batched, per_slice, gather. {REJECTION_END}"
    )
    assert warnings == []


@pytest.mark.parametrize(
    "value", ["batched", "per_slice", "gather", " Gather ", "bogus", "not a mode"]
)
def test_attached_ignores_any_copy_switch_with_one_warning(monkeypatch, warnings, value):
    """With expert DWDP the dense copies follow the expert transfer: no value is an error."""
    violation = _Violation("copy", "", env={DENSE_COPY_ENV: value})
    assert _check(monkeypatch, ATTACHED, violation) is ATTACHED
    assert warnings == [ATTACHED_COPY_WARNING.format(value=value.strip())]


# ---------------------------------------------------------------------------
# CUDA graphs, priming and drafters
# ---------------------------------------------------------------------------


def _embedded_dspark(checkpoint: Path) -> DSparkDecodingConfig:
    """A DSpark draft in the ``mtp.*`` namespace of the target checkpoint, as DeepSeek-V4-Pro's."""
    index = {"weight_map": {"mtp.0.eh_proj.weight": "model.safetensors"}}
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps(index))
    return DSparkDecodingConfig(max_draft_len=5, speculative_model=checkpoint)


#: ``speculative_config`` by name, built around a checkpoint directory; ``dspark`` is Kimi-K3's.
_DRAFTERS: dict[str, Callable[[Path], DecodingBaseConfig]] = {
    "dspark": lambda path: DSparkDecodingConfig(max_draft_len=5, speculative_model=path),
    "embedded_dspark": _embedded_dspark,
    "mtp": lambda path: MTPDecodingConfig(max_draft_len=1),
    "eagle3": lambda path: Eagle3DecodingConfig(max_draft_len=3, speculative_model=path),
    "draft_target": lambda path: DraftTargetDecodingConfig(max_draft_len=3, speculative_model=path),
}


class _GraphCase(NamedTuple):
    """A launch with CUDA graphs, a drafter or ``TRTLLM_DWDP_DENSE_PRIME``, and the verdict."""

    name: str
    mode: DenseMode
    rejection: str | None
    """A pattern the error matches; None if the gate accepts the launch."""
    drafter: str | None = None
    """A key of ``_DRAFTERS``."""
    graphs: bool = True
    """Whether the launch has ``cuda_graph_config``."""
    copy: str | None = "gather"
    """``TRTLLM_DWDP_DENSE_COPY``; None leaves it unset."""
    prime: str | None = None
    """``TRTLLM_DWDP_DENSE_PRIME``; None leaves it unset."""
    warnings: int = 1
    """How many warnings an accepted launch logs."""
    sm: int = _SM_VERSION
    """The GPU's SM version."""


#: A graph-mode drafter rejection names the one drafter accepted, then the rejected one.
_DSPARK_ONLY = (
    "DSparkDecodingConfig with draft_is_embedded_in_target=False.*, but speculative_config is "
)

_GRAPH_CASES = [
    _GraphCase("gather", STANDALONE, None),
    _GraphCase("batched", STANDALONE, None, copy="batched", warnings=0),
    _GraphCase("batched_rubin", STANDALONE, None, copy="batched", sm=107),
    _GraphCase("dspark_entry", STANDALONE, None, "dspark", prime="entry"),
    _GraphCase("eager_prime", STANDALONE, None, graphs=False, prime="x", warnings=2),
    _GraphCase("copy_unset", STANDALONE, f"but {DENSE_COPY_ENV} is unset", copy=None),
    _GraphCase("per_slice", STANDALONE, f"but {DENSE_COPY_ENV}='per_slice'", copy="per_slice"),
    _GraphCase("bad_prime", STANDALONE, f"{DENSE_PRIME_ENV}='x'", prime="x"),
    _GraphCase("mtp", STANDALONE, _DSPARK_ONLY + "MTPDecodingConfig", "mtp"),
    _GraphCase("eagle3", STANDALONE, _DSPARK_ONLY + "Eagle3DecodingConfig", "eagle3"),
    _GraphCase("draft_target", STANDALONE, _DSPARK_ONLY + "DraftTarget", "draft_target"),
    _GraphCase("embedded_dspark", STANDALONE, _DSPARK_ONLY + "an embedded", "embedded_dspark"),
    _GraphCase("attached_graphs", ATTACHED, "requires cuda_graph_config=None, but"),
    _GraphCase("eager_dspark", STANDALONE, "speculative_config=None, but", "dspark", graphs=False),
    _GraphCase("attached_dspark", ATTACHED, "speculative_config=None, but", "dspark", graphs=False),
]


@pytest.mark.parametrize("case", _GRAPH_CASES, ids=[case.name for case in _GRAPH_CASES])
def test_graph_mode_and_drafters(monkeypatch, warnings, tmp_path, case):
    """Standalone only: graphs need gather or batched set, and with them DSpark is the drafter."""
    for name, value in ((DENSE_COPY_ENV, case.copy), (DENSE_PRIME_ENV, case.prime)):
        if value is not None:
            monkeypatch.setenv(name, value)
    monkeypatch.setattr(gate, "get_sm_version", lambda: case.sm)
    llm_args = _launch(
        case.mode,
        cuda_graph_config=CudaGraphConfig() if case.graphs else None,
        speculative_config=_DRAFTERS[case.drafter](tmp_path) if case.drafter else None,
    )
    if case.rejection is not None:
        with pytest.raises(ValueError, match=case.rejection):
            validate_dense_gate(llm_args, _mapping(case.mode))
        return
    assert validate_dense_gate(llm_args, _mapping(case.mode)) is case.mode
    assert dense_graph_mode(llm_args, case.mode) is case.graphs
    assert dense_graph_mode(llm_args, None) is False
    assert len(warnings) == case.warnings


@pytest.mark.parametrize("sm", [100, 103, 107])
def test_graph_mode_batched_copies_are_logged_for_the_platform(monkeypatch, messages, sm):
    """One INFO line with the copy engines; on Rubin (SM107) one warning recommending the gather."""
    monkeypatch.setenv(DENSE_COPY_ENV, "batched")
    monkeypatch.setattr(gate, "get_sm_version", lambda: sm)
    llm_args = _launch(STANDALONE, cuda_graph_config=CudaGraphConfig())

    assert validate_dense_gate(llm_args, _mapping(STANDALONE)) is STANDALONE

    level, message = messages[0]
    if sm == 107:
        assert (level, message) == ("warning", BATCHED_GRAPH_RUBIN_WARNING)
    else:
        assert (level, message) == ("info", BATCHED_GRAPH_INFO)
    assert [level for level, _ in messages] == [level, "info"]  # then the gate's own line
    assert "graph mode on" in messages[-1][1]


def test_the_copy_engine_line_names_a_pdl_break_turned_on(monkeypatch, messages):
    monkeypatch.setenv(DENSE_COPY_ENV, "batched")
    monkeypatch.setenv(DENSE_PDL_BREAK_ENV, "1")
    llm_args = _launch(STANDALONE, cuda_graph_config=CudaGraphConfig())

    assert validate_dense_gate(llm_args, _mapping(STANDALONE)) is STANDALONE

    assert messages[0] == (
        "info",
        BATCHED_GRAPH_INFO.replace("is off", f"is on ({DENSE_PDL_BREAK_ENV}=1)"),
    )


@pytest.mark.parametrize(
    "copy, graphs, probed",
    [("batched", True, True), ("batched", False, False), ("gather", True, False)],
    ids=["batched_graphs", "batched_eager", "gather_graphs"],
)
def test_only_graph_mode_batched_probes_the_capture_of_batched_copies(
    monkeypatch, _gate_host, copy, graphs, probed
):
    monkeypatch.setenv(DENSE_COPY_ENV, copy)
    llm_args = _launch(STANDALONE, cuda_graph_config=CudaGraphConfig() if graphs else None)

    assert validate_dense_gate(llm_args, _mapping(STANDALONE)) is STANDALONE

    assert _gate_host == (["probe"] if probed else [])


def test_graph_mode_batched_needs_a_driver_that_captures_batched_copies(monkeypatch, messages):
    """A driver that cannot capture ``cudaMemcpyBatchAsync`` fails the launch, naming the gather."""
    failure = (
        "a CUDA graph capture of cudaMemcpyBatchAsync failed: operation not permitted when stream "
        "is capturing (CUDA driver 13.0, CUDA runtime 13.0)"
    )
    monkeypatch.setattr(gate, "batched_copy_capture_error", lambda: failure)
    monkeypatch.setenv(DENSE_COPY_ENV, "batched")
    llm_args = _launch(STANDALONE, cuda_graph_config=CudaGraphConfig())

    with pytest.raises(ValueError) as error:
        validate_dense_gate(llm_args, _mapping(STANDALONE))

    assert str(error.value) == (
        f"Dense DWDP ({DENSE_ENV}) requires a CUDA driver that captures cudaMemcpyBatchAsync in a "
        f"CUDA graph for {DENSE_COPY_ENV}=batched with cuda_graph_config, but {failure}; set "
        f"{DENSE_COPY_ENV}=gather, whose SM gather kernel the decode CUDA graphs capture instead. "
        f"{REJECTION_END}"
    )
    assert messages == []


# ---------------------------------------------------------------------------
# Generation-server shapes
# ---------------------------------------------------------------------------


def _unset_gather_defaults(monkeypatch) -> None:
    """Unset the gather's compute-side variables; the test's teardown unsets them again."""
    for name in _GATHER_DEFAULTS:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def _generation_server(checkpoint: Path, **overrides) -> TorchLlmArgs:
    """A Kimi-K3 generation server as GB300 runs it with dense DWDP: DEP16 (tp 16, attention DP,
    MoE EP 16, cp 1) on ``MEGAMOE_CUTEDSL``, the DSpark drafter with its own checkpoint, decode
    CUDA graphs with padding, the overlap scheduler and an FP8 KV cache on the KV cache manager
    v2; with ``overrides`` applied unvalidated."""
    llm_args = _launch(
        STANDALONE,
        disable_overlap_scheduler=False,
        cuda_graph_config=CudaGraphConfig(batch_sizes=[1, 2, 4, 8, 16, 32], enable_padding=True),
        moe_config=MoeConfig(backend="MEGAMOE_CUTEDSL", use_low_precision_moe_combine=True),
        kv_cache_config=KvCacheConfig(
            dtype="fp8",
            tokens_per_block=64,
            use_kv_cache_manager_v2=True,
            enable_block_reuse=False,
            host_cache_size=0,
        ),
        speculative_config=DSparkDecodingConfig(max_draft_len=5, speculative_model=checkpoint),
        max_batch_size=32,
        max_num_tokens=32 * 6,
    )
    return llm_args.model_copy(update=overrides)


def test_the_gb300_generation_server_runs_standalone_in_graph_mode_without_an_sm_reserve(
    monkeypatch, messages, tmp_path
):
    """``TRTLLM_DWDP_DENSE_COPY=batched``: standalone, graph mode, copy engines, no SM reserve.

    The creator applies the gather's compute-side defaults (the SM reserve among them) before the
    model is built only for the gather, so with the copy engines none of them is set, and setup
    leaves the PDL break off.
    """
    monkeypatch.setenv(DENSE_COPY_ENV, "batched")
    _unset_gather_defaults(monkeypatch)
    llm_args = _generation_server(tmp_path)
    mapping = _mapping(STANDALONE)
    assert (mapping.tp_size, mapping.moe_ep_size, mapping.cp_size) == (16, 16, 1)

    assert validate_dense_gate(llm_args, mapping) is STANDALONE

    assert dense_graph_mode(llm_args, STANDALONE)
    assert messages[0] == ("info", BATCHED_GRAPH_INFO)
    assert not any(level == "warning" for level, _ in messages)
    assert py_executor_creator._reserve_dense_gather_sms(llm_args) is False
    assert [name for name in _GATHER_DEFAULTS if name in os.environ] == []
    assert not DenseConfig.from_env().pdl_break_enabled(use_gather=False, closed=True)


def test_the_same_server_with_the_gather_reserves_its_sms_before_the_model_is_built(
    monkeypatch, tmp_path
):
    """The negative of the GB300 case: ``gather`` with MegaMoE sets the reserve up front."""
    monkeypatch.setenv(DENSE_COPY_ENV, "gather")
    _unset_gather_defaults(monkeypatch)
    llm_args = _generation_server(tmp_path)

    assert validate_dense_gate(llm_args, _mapping(STANDALONE)) is STANDALONE

    assert py_executor_creator._reserve_dense_gather_sms(llm_args) is True
    assert [name for name in _GATHER_DEFAULTS if name in os.environ] == list(_GATHER_DEFAULTS)


@pytest.mark.parametrize(
    "mapping, actual",
    [
        (
            dict(tp_size=1, cp_size=16, enable_attention_dp=False),
            "but this launch has no dwdp_config and enable_attention_dp=False.",
        ),
        (
            dict(tp_size=2, cp_size=8, enable_attention_dp=True),
            "requires cp_size=1, but cp_size=8.",
        ),
    ],
    ids=["helix16", "helix8_attention_dp"],
)
def test_a_helix_generation_server_is_rejected(monkeypatch, tmp_path, mapping, actual):
    """Helix context parallelism splits the attention over the ranks: there is no group of ranks
    that each hold every layer whole, as dense DWDP shards them."""
    monkeypatch.setenv(DENSE_COPY_ENV, "batched")
    helix = Mapping(
        world_size=16,
        rank=0,
        moe_ep_size=16,
        cp_config={"cp_type": CpType.HELIX},
        **mapping,
    )
    with pytest.raises(ValueError, match=re.escape(actual)):
        validate_dense_gate(_generation_server(tmp_path), helix)
