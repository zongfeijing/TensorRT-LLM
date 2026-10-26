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
"""Dense DWDP launch gate: reject what dense DWDP cannot serve before the model exists.

``py_executor_creator`` calls :func:`validate_dense_gate` next to its DWDP checks, before it
constructs any DWDP or dense DWDP manager and the model, so an unsupported launch fails before the
weights load rather than in setup or in the first forward.  The gate also decides how dense DWDP
runs: with expert DWDP (attached) or standalone on an attention-DP instance, and standalone with
CUDA graphs (graph mode, :func:`dense_graph_mode`) or without.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, NoReturn

from tensorrt_llm._utils import get_sm_version, mpi_disabled
from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig, PrefillCudaGraphBackend
from tensorrt_llm.logger import logger

from .. import fabric_gather, vmm
from .config import DENSE_COPY_ENV, DENSE_ENV, DENSE_PDL_BREAK_ENV, DENSE_PRIME_ENV, DenseConfig
from .plan import batched_copy_capture_error
from .types import LOG_PREFIX, CopyMode, DenseMode, PrimePolicy

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig, TorchLlmArgs
    from tensorrt_llm.mapping import Mapping

#: The two launch shapes dense DWDP runs in; every mode rejection names both.
_MODES = (
    "runs either with expert DWDP (dwdp_config with dwdp_size > 1, tp_size=1, "
    "disable_overlap_scheduler=True) or standalone on an attention-DP instance (no dwdp_config, "
    "enable_attention_dp=True, tp_size > 1)"
)

#: The speculative decoding graph mode serves; every graph-mode speculation rejection names it.
_SPECULATION = (
    "speculative_config=None or the DSpark drafter with a checkpoint of its own "
    "(DSparkDecodingConfig with draft_is_embedded_in_target=False)"
)

#: The server role ``trtllm-serve`` gives a worker; logged, not checked.
_DISAGG_ROLE_ENV = "TRTLLM_DISAGG_ROLE"


def validate_dense_gate(llm_args: TorchLlmArgs, mapping: Mapping) -> DenseMode:
    """Reject launches dense DWDP cannot serve, before the model is built; return the mode.

    Called only when ``dense_dwdp_requested()``.  The checks, in this order:

    1. Valid ``TRTLLM_DWDP_DENSE*`` switches: ``DenseConfig.from_env()`` reads them all, so an
       invalid ``TRTLLM_DWDP_DENSE_SEGMENTS`` fails here rather than in setup P1, and the
       warnings about values it ignores are logged before the load as well.
       ``TRTLLM_DWDP_DENSE_COPY`` and ``TRTLLM_DWDP_DENSE_PRIME`` are read but not checked here
       (step 3).
    2. The mode.  With ``dwdp_config``, attached: ``dwdp_size > 1`` (the dense rows are sharded
       over the DWDP group), ``tp_size == 1`` and the overlap scheduler off, as expert DWDP
       requires.  Without, standalone: ``enable_attention_dp`` and ``tp_size > 1``; every rank of
       the instance then holds the same dense weights, so the rows are sharded over the
       attention-DP (tp) group, and the overlap scheduler may be on.  Any other launch is
       rejected with a message that names both modes, including expert DWDP over an
       attention-DP instance (``dwdp_config`` with ``tp_size > 1``), which dense DWDP does
       not run with.
    3. ``TRTLLM_DWDP_DENSE_COPY``: standalone it must be valid
       (``DenseConfig.standalone_copy_mode``); with expert DWDP the dense copies follow the
       expert transfer, so any value, valid or not, is ignored with one warning.
       ``TRTLLM_DWDP_DENSE_PRIME``: in graph mode (:func:`dense_graph_mode`) it must be valid
       (``DenseConfig.prime_policy``); outside it dense DWDP primes at the start of every
       forward, so any value, valid or not, is ignored with one warning.
    4. In both modes:

       * ``cuda_graph_config`` None with expert DWDP, whose per-layer hooks need eager
         single-stream execution: they keep host-side slot state and order the copies and the
         weight reads with stream events in every forward.  Standalone it turns on graph mode,
         which captures the dense schedule of a whole step into every decode CUDA graph; then
         ``TRTLLM_DWDP_DENSE_COPY`` must be set to ``gather`` or ``batched``, so that a
         generation server always runs the transfer chosen for its platform (step 5), never the
         default unawares, and ``per_slice`` is rejected.
       * ``torch_compile_config`` None and ``prefill_cuda_graph_backend`` disabled: the hooks
         run eagerly or inside the decode CUDA graphs, never traced or captured otherwise.
       * ``sleep_config`` None: the moved weights live in the staging slots and the row store,
         outside the memory pools that sleep and wake-up manage.
       * ``lora_config`` None: it reads layer weights outside the layer's read window.
       * ``speculative_config`` None.  In graph mode it may also be the DSpark drafter with a
         checkpoint of its own (``DSparkDecodingConfig`` with ``draft_is_embedded_in_target``
         False), which reads only its own weights and the target's ``lm_head``, none of them
         moved, inside the target's forward.  MTP, Eagle, draft-target, an embedded DSpark draft
         and every other drafter are rejected: they may read layer weights outside the layer's
         read window.
       * ``enable_locality_domains`` off: locality domains make ``Linear`` and ``MLA`` keep
         ``_locality_domain_*`` shard copies of their weights, which dense DWDP does not move,
         so they would stay replicated.
       * ``mapping.pp_size`` and ``cp_size`` 1: every rank of the group holds every layer.
       * FABRIC peer handles (``vmm.peer_handle_type()``): the row store is shared only through
         FABRIC handles.
    5. Standalone only: the MPI backend (``TLLM_DISABLE_MPI`` unset), since the dense group is an
       MPI sub-communicator over the attention-DP group; with ``TRTLLM_DWDP_DENSE_COPY=gather``
       a registered ``trtllm::dwdp_fabric_gather``; and in graph mode with ``batched`` a CUDA
       driver that captures ``cudaMemcpyBatchAsync`` in a CUDA graph
       (``plan.batched_copy_capture_error``: one small batched copy captured into a throwaway
       graph and replayed), so that a driver that cannot fails the launch here, with its
       versions and ``gather`` as the way out, not in the first decode graph capture.  A
       standalone gather launch that passes logs one warning, which names the settings the gather
       was validated with on a context server or, in graph mode, the SM reserve set for it.  A
       graph-mode ``batched`` launch logs one INFO line: the copy engines hold no SMs, so no SM
       reserve applies and the PDL break (``TRTLLM_DWDP_DENSE_PDL_BREAK``) is off unless set.  On
       Rubin (SM107), where generation servers were validated with the SM gather, it logs one
       warning that recommends ``gather`` instead.

    A launch that passes logs its mode, graph mode, priming policy and ``TRTLLM_DISAGG_ROLE``.

    Args:
        llm_args: The launch configuration.
        mapping: The rank mapping, with the DWDP fields filled in.

    Returns:
        ``DenseMode.ATTACHED`` with ``dwdp_config``, else ``DenseMode.STANDALONE``.

    Raises:
        ValueError: Naming the first violated requirement.
    """
    try:
        config = DenseConfig.from_env()
    except ValueError as exc:
        _reject(f"valid {DENSE_ENV}_* switches", str(exc))
    mode = _launch_mode(llm_args, mapping)
    graph_mode = dense_graph_mode(llm_args, mode)
    copy_mode = None
    if mode is DenseMode.STANDALONE:
        try:
            copy_mode = config.standalone_copy_mode()
        except ValueError as exc:
            _reject(f"valid {DENSE_ENV}_* switches", str(exc))
    elif config.copy is not None:
        logger.warning(
            f"{LOG_PREFIX} {DENSE_COPY_ENV}={config.copy!r} is ignored with expert DWDP: the "
            f"dense copies run on the expert DWDP copy stream with the expert transfer"
        )
    prime = _prime_policy(config, graph_mode)
    if graph_mode:
        _check_graph_copy_mode(config, copy_mode)
    else:
        _require_unset("cuda_graph_config", llm_args.cuda_graph_config)
    _require_unset("torch_compile_config", llm_args.torch_compile_config)
    prefill_graphs = llm_args.prefill_cuda_graph_backend
    if prefill_graphs != PrefillCudaGraphBackend.DISABLED:
        _reject(
            f"prefill_cuda_graph_backend='{PrefillCudaGraphBackend.DISABLED}'",
            f"prefill_cuda_graph_backend='{prefill_graphs}'",
        )
    _require_unset("sleep_config", llm_args.sleep_config)
    _require_unset("lora_config", llm_args.lora_config)
    _check_speculation(llm_args.speculative_config, graph_mode)
    if llm_args.enable_locality_domains:
        _reject("enable_locality_domains=False", "enable_locality_domains=True")
    for name in ("pp_size", "cp_size"):
        size = getattr(mapping, name)
        if size != 1:
            _reject(f"{name}=1", f"{name}={size}")
    handle_type = vmm.peer_handle_type()
    if handle_type != vmm.cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC:
        _reject(
            "FABRIC peer handles (CU_MEM_HANDLE_TYPE_FABRIC)",
            f"this host shares {handle_type.name} handles",
        )
    if mode is DenseMode.STANDALONE:
        _check_standalone(config, copy_mode, graph_mode)
    role = os.environ.get(_DISAGG_ROLE_ENV) or "unset"
    logger.info(
        f"{LOG_PREFIX} launch gate passed: {mode.value} mode, graph mode "
        f"{'on' if graph_mode else 'off'}, {prime.value} priming, {_DISAGG_ROLE_ENV}={role}"
    )
    return mode


def dense_graph_mode(llm_args: TorchLlmArgs, mode: DenseMode | None) -> bool:
    """Whether dense DWDP runs in graph mode: standalone, with ``cuda_graph_config``.

    Graph mode runs the ``CLOSED`` protocol, which captures the dense schedule of a whole step
    into every decode CUDA graph, as a generation server needs.  Attached dense DWDP, for which
    the gate rejects CUDA graphs, and standalone dense DWDP without them run the ``LEGACY``
    protocol.  ``py_executor_creator`` passes the result to ``DenseDwdpManager``.

    Args:
        llm_args: The launch configuration.
        mode: What :func:`validate_dense_gate` returned; None when dense DWDP is not requested.
    """
    return mode is DenseMode.STANDALONE and llm_args.cuda_graph_config is not None


def _launch_mode(llm_args: TorchLlmArgs, mapping: Mapping) -> DenseMode:
    """The mode the launch asks for (step 2); rejects a launch that fits neither."""
    dwdp_config = llm_args.dwdp_config
    if dwdp_config is not None:
        if dwdp_config.dwdp_size <= 1:
            _reject_mode(f"dwdp_config with dwdp_size={dwdp_config.dwdp_size}")
        if mapping.tp_size != 1:
            _reject_mode(f"dwdp_config with tp_size={mapping.tp_size}")
        if not llm_args.disable_overlap_scheduler:
            _reject_mode("dwdp_config with disable_overlap_scheduler=False")
        return DenseMode.ATTACHED
    if not mapping.enable_attention_dp:
        _reject_mode("no dwdp_config and enable_attention_dp=False")
    if mapping.tp_size <= 1:
        _reject_mode(f"no dwdp_config and tp_size={mapping.tp_size}")
    return DenseMode.STANDALONE


def _prime_policy(config: DenseConfig, graph_mode: bool) -> PrimePolicy:
    """The priming policy (step 3): the switch's in graph mode, else ``ENTRY`` whatever it says."""
    if graph_mode:
        try:
            return config.prime_policy()
        except ValueError as exc:
            _reject(f"valid {DENSE_ENV}_* switches", str(exc))
    if config.prime is not None:
        logger.warning(
            f"{LOG_PREFIX} {DENSE_PRIME_ENV}={config.prime!r} is ignored: it applies to "
            f"standalone dense DWDP with cuda_graph_config, and this launch primes the first two "
            f"covered layers at the start of every forward"
        )
    return PrimePolicy.ENTRY


def _check_graph_copy_mode(config: DenseConfig, copy_mode: CopyMode) -> None:
    """Graph mode's transfer (step 4): set explicitly, to ``gather`` or ``batched``."""
    requirement = f"{DENSE_COPY_ENV} set to gather or batched with cuda_graph_config"
    if config.copy is None:
        _reject(requirement, f"{DENSE_COPY_ENV} is unset")
    if copy_mode is CopyMode.PER_SLICE:
        _reject(requirement, f"{DENSE_COPY_ENV}={config.copy!r}")


def _check_speculation(speculative_config: DecodingBaseConfig | None, graph_mode: bool) -> None:
    """``speculative_config`` (step 4): None, or in graph mode also the DSpark drafter."""
    if not graph_mode:
        _require_unset("speculative_config", speculative_config)
        return
    if speculative_config is None:
        return
    if isinstance(speculative_config, DSparkDecodingConfig):
        if not speculative_config.draft_is_embedded_in_target:
            return
        actual = "an embedded DSpark draft (draft_is_embedded_in_target=True)"
    else:
        actual = type(speculative_config).__name__
    _reject(_SPECULATION, f"speculative_config is {actual}")


def _check_standalone(config: DenseConfig, copy_mode: CopyMode, graph_mode: bool) -> None:
    """The standalone-only requirements (step 5), after the common ones."""
    if mpi_disabled():
        _reject(
            "the MPI backend (the standalone dense group is an MPI sub-communicator over the "
            "attention-DP group)",
            "TLLM_DISABLE_MPI=1",
        )
    if copy_mode is CopyMode.BATCHED and graph_mode:
        _check_batched_graph_copies(config)
    if copy_mode is not CopyMode.GATHER:
        return
    if fabric_gather.gather_op() is None:
        _reject(
            f"trtllm::dwdp_fabric_gather for {DENSE_COPY_ENV}=gather",
            "this build does not register it",
        )
    if graph_mode:
        logger.warning(
            f"{LOG_PREFIX} {DENSE_COPY_ENV}=gather with CUDA graphs: every step gathers every "
            f"covered layer, and the gather kernel holds SMs next to the decode kernels. Unless "
            f"set explicitly, TRTLLM_CUTEDSL_SM_RESERVE keeps the gather's SMs out of the CuTeDSL "
            f"persistent grids; with MEGAMOE_CUTEDSL it is set before the model is built, as "
            f"MegaMoE sizes its grid then"
        )
        return
    logger.warning(
        f"{LOG_PREFIX} {DENSE_COPY_ENV}=gather without expert DWDP: the gather kernel holds SMs "
        f"next to the MoE all-to-all. It was validated on a Rubin DEP16 context server with "
        f"TRTLLM_DWDP_GATHER_KERNEL=tma and TLLM_ROUTING_DISABLE_COOP_LAUNCH=1, the latter of "
        f"which the setup applies unless it is set: cooperative routing waits for the SMs the "
        f"gather holds"
    )


def _check_batched_graph_copies(config: DenseConfig) -> None:
    """Graph mode with ``batched`` (step 5): the capture probe, then the transfer's log line."""
    error = batched_copy_capture_error()
    if error is not None:
        _reject(
            f"a CUDA driver that captures cudaMemcpyBatchAsync in a CUDA graph for "
            f"{DENSE_COPY_ENV}=batched with cuda_graph_config",
            f"{error}; set {DENSE_COPY_ENV}=gather, whose SM gather kernel the decode CUDA "
            f"graphs capture instead",
        )
    if get_sm_version() == 107:
        logger.warning(
            f"{LOG_PREFIX} {DENSE_COPY_ENV}=batched with CUDA graphs on Rubin (SM107): every step "
            f"copies every covered layer; generation servers were validated on this GPU with "
            f"{DENSE_COPY_ENV}=gather, which is recommended here"
        )
        return
    if config.pdl_break_enabled(use_gather=False, closed=True):
        pdl_break = f"the PDL break is on ({DENSE_PDL_BREAK_ENV}=1)"
    else:
        pdl_break = "the PDL break is off"
    logger.info(
        f"{LOG_PREFIX} {DENSE_COPY_ENV}=batched with CUDA graphs: every step copies every covered "
        f"layer on the copy engines, which hold no SMs, so no SM reserve applies and {pdl_break}"
    )


def _require_unset(name: str, value: object) -> None:
    if value is not None:
        _reject(f"{name}=None", f"{name} is set")


def _reject(requirement: str, actual: str) -> NoReturn:
    raise ValueError(
        f"Dense DWDP ({DENSE_ENV}) requires {requirement}, but {actual}. Change the launch "
        f"configuration or unset {DENSE_ENV}."
    )


def _reject_mode(actual: str) -> NoReturn:
    raise ValueError(
        f"Dense DWDP ({DENSE_ENV}) {_MODES}, but this launch has {actual}. Change the launch "
        f"configuration or unset {DENSE_ENV}."
    )
