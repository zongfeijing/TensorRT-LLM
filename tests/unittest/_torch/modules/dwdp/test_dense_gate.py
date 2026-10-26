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
  ``enable_attention_dp`` and ``tp_size > 1``; the overlap scheduler may be on.

It checks, in order: the switches; the mode (a launch that fits neither is rejected with a
message naming both); ``TRTLLM_DWDP_DENSE_COPY`` (standalone: it must be valid; attached: any
value is ignored with one warning); the requirements both modes share; and, standalone only, the
MPI backend and, for ``gather``, the gather op, after which it warns that the gather is
experimental.

Every test starts without ``TRTLLM_DWDP_DENSE*`` variables, with MPI on, FABRIC peer handles and
a registered gather op; a test changes only what it checks.
"""

import os
import re
from types import SimpleNamespace
from typing import NamedTuple

import pytest

from tensorrt_llm._torch.modules.dwdp import fabric_gather, vmm
from tensorrt_llm._torch.modules.dwdp.dense import gate
from tensorrt_llm._torch.modules.dwdp.dense.config import DENSE_COPY_ENV, DENSE_ENV
from tensorrt_llm._torch.modules.dwdp.dense.gate import validate_dense_gate
from tensorrt_llm._torch.modules.dwdp.dense.types import DenseMode
from tensorrt_llm.llmapi.llm_args import DwdpConfig, TorchLlmArgs
from tensorrt_llm.mapping import Mapping

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


@pytest.fixture(autouse=True)
def _gate_host(monkeypatch):
    """No ``TRTLLM_DWDP_DENSE*`` switch, MPI on, FABRIC peer handles, a registered gather op."""
    for name in [name for name in os.environ if name.startswith(DENSE_ENV)]:
        monkeypatch.delenv(name)
    monkeypatch.setattr(gate, "mpi_disabled", lambda: False)
    monkeypatch.setattr(vmm, "peer_handle_type", lambda: _HANDLE_TYPE.CU_MEM_HANDLE_TYPE_FABRIC)
    monkeypatch.setattr(fabric_gather, "gather_op", lambda: _GATHER_OP)


@pytest.fixture
def warnings(monkeypatch) -> list[str]:
    """The gate's own warnings, in order; the switch warnings of ``config`` are not among them."""
    messages = []
    monkeypatch.setattr(gate, "logger", SimpleNamespace(warning=messages.append))
    return messages


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
