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
"""Dense DWDP in CUDA graph mode on real Kimi-K3 modules: decode and MTP verify are bit-exact.

It runs on one Blackwell or Rubin GPU (SM100, SM103 or SM107), on the nine-layer Kimi-K3 model
of ``test_dense_k3_modules_gpu`` (S7), whose helpers build it, load its random checkpoint through
the real load path and set dense DWDP up.  Here the model has a generation server's caches:
``MambaHybridCacheManagerV2``, built as the executor builds it for Kimi-K3 with a DSpark config of
five drafts, holds the KDA states in BF16 (``mamba_ssm_cache_dtype``, as a generation server sets
it) with the replay caches of the fused verify kernel, and the MLA latent cache, in BF16 or in
FP8, with 64-token pages.

Every request is prefilled once.  Each later forward is a decode (one token per request) or an
MTP verify (one golden and five draft tokens per request) of the whole batch, from the KDA states
the prefill left: the test restores them before each one, and each rewrites the KV positions past
the prefill that it reads, so forwards of one kind compute the same.  The KDA layers run the fused
decode kernel and the fused verify kernel, which replays from the replay caches (no accepted draft
is pending, so it replays none); the MLA layers run absorbed generation, whose absorption weights
only generation reads.  With 4 requests every KDA projection has at most 128 rows and forks its
``[f_a | b]`` chain onto the auxiliary stream; with 24 the verify has 144 rows and does not.

The same forwards run twice: on the replicated weights, then on dense DWDP's slots, standalone in
graph mode (the ``CLOSED`` protocol with wrap priming) on rank 12 of a 16-rank loopback group,
``TRTLLM_DWDP_DENSE_STRICT`` and ``TRTLLM_DWDP_DENSE_DEBUG=2``, with either transfer: the TMA SM
gather with its SM reserve and GEMM scheduler (``TRTLLM_DWDP_DENSE_COPY=gather``, as on Rubin), or
one batched copy-engine copy per layer without them (``batched``, as on GB300).  They are the
causal LM's own forwards, so ``KimiLinearForCausalLM.forward`` opens the dense forward scope (H0
and H4) around the text model and the LM head: eager, with multi-stream execution on (the warmup
before a capture) and off (an eager fallback step), and captured into a decode graph and a verify
graph with it on, as ``CUDAGraphRunner`` captures them.  The replays of both graphs and the eager
forwards follow each other on one execution stream without a host synchronization.  Spin kernels
delay the gathers of some forwards, and the read windows of others together with the reads of
dense weights on auxiliary streams by the shared expert and the KDA forget-gate projection; a
capture bakes its delays into its graph.  A read or a copy that does not wait for its event,
within a forward or across two, sees or overwrites another layer's bytes.

Pass: every layer's output, the final hidden state and the logits of every forward equal the
replicated run's bit for bit, and already there a replay equals the eager forward of its kind; no
demand fetch, fence fallback or recovery; the slot watchdog, which a replay runs as captured,
finds no mismatch; the GEMMs and their tactics are the same in both runs
(``TLLM_AUTOTUNER_CACHE_PATH`` pins them as in S7; on SM107 they include the MXFP8 weight-read
GEMM).  S7's audit of the moved bytes, the slot sharing and the released bindings, which knows the
SM107 weight layout, runs on SM107; elsewhere setup is checked without it.
"""

from __future__ import annotations

import contextlib
import functools
import weakref
from collections.abc import Iterator
from typing import NamedTuple

import pytest
import test_dense_k3_modules_gpu as k3_modules
import torch
from test_dense_k3_modules_gpu import (
    _COVERED,
    _CUT_DOWN,
    _DEPLOYMENT_ENV,
    _SPIN_CYCLES,
    _UNSET_ENV,
    ATTN,
    MOE,
    _build_model,
    _capture_output,
    _check_captured,
    _check_detached,
    _check_setup,
    _check_tactics,
    _DenseRun,
    _dep_mapping,
    _differences,
    _report,
    _require_gather,
    _set_up_dense,
    _slow_windows,
    _spin,
    _SpinningLauncher,
    _TacticLog,
    _Variant,
)

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata, KVCacheParams
from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
from tensorrt_llm._torch.models import modeling_kimi_linear as mkl
from tensorrt_llm._torch.modules.dwdp.dense.manager import DenseDwdpManager
from tensorrt_llm._torch.modules.dwdp.dense.types import (
    CopyMode,
    DenseMode,
    DenseProtocol,
    PrimePolicy,
)
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
from tensorrt_llm._torch.pyexecutor._util import _create_kv_cache_manager
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm._torch.utils import model_extra_attrs
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig, KvCacheConfig
from tensorrt_llm.models.modeling_utils import QuantAlgo

# S7's fixtures: the GEMM tactic log, and no dense DWDP manager registered around a test.
tactics = k3_modules.tactics
_no_registered_managers = k3_modules._no_registered_managers

#: The Kimi-K3 model switch of S7's deployment environment, which every case sets.
_MODEL_ENV = {"KIMI_K3_FP8_WEIGHT_READ_MOE_MLP": "1"}

#: The rest of S7's deployment environment, which belongs to the SM gather: its TMA kernel and its
#: compute-side settings (the SM reserve and the CLC dynamic GEMM scheduler), with the
#: non-cooperative routing that setup also sets for it (``apply_compute_side_defaults``).
_GATHER_ENV = {
    **{name: value for name, value in _DEPLOYMENT_ENV.items() if name not in _MODEL_ENV},
    "TLLM_ROUTING_DISABLE_COOP_LAUNCH": "1",
}

#: The GPUs that run the Kimi-K3 modules here: Blackwell (SM100, SM103) and Rubin (SM107).
_CAPABILITIES = ((10, 0), (10, 3), (10, 7))

#: KV-cache quantization of the MLA layers, and the KV cache's dtype, per KV-cache dtype.
_KV_CACHE_QUANT = {"bf16": None, "fp8": QuantAlgo.FP8}
_KV_CACHE_DTYPE = {"bf16": DataType.BF16, "fp8": DataType.FP8}


def _rubin() -> bool:
    """Whether this is a Rubin GPU, which runs the MXFP8 weight-read GEMM that S7 audits."""
    return torch.cuda.get_device_capability() == (10, 7)


@pytest.fixture(scope="module")
def device() -> Iterator[torch.device]:
    """A Blackwell or Rubin GPU with the deployment's model environment; skips otherwise.

    Unlike S7's fixture it needs neither SM107 nor the gather op: on SM100 and SM103 the FP8
    weight-read projections run DeepGEMM rather than the Rubin MXFP8 GEMM, and the batched copies
    need no gather.  The gather cases add the gather's environment and require its op
    (:func:`_transfer_env`).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    capability = torch.cuda.get_device_capability()
    if capability not in _CAPABILITIES:
        pytest.skip(f"the Kimi-K3 modules run on SM100, SM103 and SM107, not on {capability}")
    if capability == (10, 7) and not IS_CUTLASS_DSL_RUBIN_AVAILABLE:
        pytest.skip("the Kimi-K3 MXFP8 weight-read GEMM on SM107 needs CuTe DSL for Rubin")
    with pytest.MonkeyPatch.context() as patch:
        for name, value in _MODEL_ENV.items():
            patch.setenv(name, value)
        for name in (*_UNSET_ENV, *_GATHER_ENV):
            patch.delenv(name, raising=False)
        yield torch.device("cuda", torch.cuda.current_device())


def _transfer_env(
    monkeypatch: pytest.MonkeyPatch, transfer: CopyMode, device: torch.device
) -> None:
    """With the SM gather, its environment and its op for this GPU; with the copy engines, none."""
    if transfer is CopyMode.GATHER:
        for name, value in _GATHER_ENV.items():
            monkeypatch.setenv(name, value)
        _require_gather(device)


#: Prompt tokens of every request.
_PREFILL = 40

#: Draft tokens per request of an MTP verify (a DSpark drafter's block of five drafts).
_DRAFTS = 5

#: Tokens per request of each kind of forward.
_TOKENS = {"prefill": _PREFILL, "decode": 1, "verify": 1 + _DRAFTS}


def _gen_variant(transfer: CopyMode) -> _Variant:
    """Rank 12 of a DEP16 generation instance with ``transfer``, with the slot watchdog."""
    return _Variant((ATTN, MOE), rank=12, transfer=transfer, debug=2, mode=DenseMode.STANDALONE)


class _Step(NamedTuple):
    """One forward of a test."""

    kind: str
    """``prefill``, ``decode`` or ``verify``."""

    mode: str
    """``eager``; ``capture``, into the kind's graph, which runs nothing; or ``replay``."""

    delay: str = ""
    """With dense DWDP: ``copies`` spins before every gather, ``reads`` before every read window
    and, on their auxiliary streams, before the shared expert and the KDA forget-gate projection."""

    multi_stream: bool = True
    """Multi-stream execution: on in a capture and the warmup before it, off in eager fallbacks."""


#: The forwards of a test, in order.  Eager forwards and replays follow each other both ways, so
#: a forward's copies must wait for the reads of the forward before it, and its first reads for the
#: copies that forward enqueued last (the wrap gathers of the next forward's first layers).
_STEPS = (
    _Step("prefill", "eager", multi_stream=False),
    _Step("decode", "eager", "copies"),
    _Step("decode", "capture", "reads"),
    _Step("decode", "replay"),
    _Step("verify", "eager", "reads", multi_stream=False),
    _Step("verify", "capture", "copies"),
    _Step("verify", "replay"),
    _Step("decode", "replay"),
    _Step("decode", "eager", "copies", multi_stream=False),
    _Step("verify", "replay"),
    _Step("verify", "eager", "reads"),
    _Step("decode", "replay"),
)


class _JoiningRoutedExperts(k3_modules._RoutedExpertsStandIn):
    """S7's routed-expert stand-in, joining the latent projection's stream as the experts do.

    With multi-stream execution and the CUTEDSL backend, ``KimiK3MoERuntime`` runs the latent down
    projection on the ``MoeOutputMemset`` stream and passes its ready event to the routed experts,
    which wait for it before they read their input.
    """

    def forward(self, x, router_logits, *, input_ready_event=None, **kwargs) -> torch.Tensor:
        if input_ready_event is not None:
            input_ready_event.wait()
        return super().forward(x, router_logits, **kwargs)


def _build_gen_model(
    kv_cache: str, monkeypatch: pytest.MonkeyPatch, device: torch.device
) -> mkl.KimiLinearForCausalLM:
    """S7's model with the generation server's cache dtypes and routed experts that join.

    The KV-cache dtype is set as the model loader sets ``kv_cache_config.dtype``: on the model's
    quantization config and on every per-layer one.
    """
    model_config = k3_modules._model_config

    def gen_model_config(cfg, mapping=None):
        config = model_config(cfg, mapping)
        config.quant_config.mamba_ssm_cache_dtype = "bfloat16"
        config.quant_config.kv_cache_quant_algo = _KV_CACHE_QUANT[kv_cache]
        for layer_quant_config in (config.quant_config_dict or {}).values():
            layer_quant_config.kv_cache_quant_algo = _KV_CACHE_QUANT[kv_cache]
        return config

    monkeypatch.setattr(k3_modules, "_model_config", gen_model_config)
    monkeypatch.setattr(k3_modules, "_RoutedExpertsStandIn", _JoiningRoutedExperts)
    return _build_model(monkeypatch, device)


class _GenCache:
    """A generation server's caches for ``batch`` requests, and the metadata of each forward kind.

    The requests are added as the warmup adds its dummies, with room for the prefill and a verify.
    The decode and verify metadata are CUDA graph metadata, which the eager forwards use too, as
    the warmup before a capture does; the verify metadata has the speculative parameters the
    executor sets.

    Args:
        model: The model the caches serve.
        batch: Requests, each in its own KDA state slot.
        device: The CUDA device.
    """

    def __init__(self, model: mkl.KimiLinearForCausalLM, batch: int, device: torch.device) -> None:
        config = model.model_config
        self.manager = _create_kv_cache_manager(
            None,
            MambaHybridCacheManagerV2,
            config.mapping,
            KvCacheConfig(
                max_tokens=batch * 1024,
                enable_block_reuse=False,
                host_cache_size=0,
                use_kv_cache_manager_v2=True,
            ),
            tokens_per_block=64,
            max_seq_len=1024,
            max_batch_size=batch,
            spec_config=DSparkDecodingConfig(max_draft_len=_DRAFTS),
            sparse_attention_config=None,
            max_num_tokens=batch * _PREFILL,
            max_beam_width=1,
            kv_connector_manager=None,
            model_config=config,
            dtype=torch.bfloat16,
            is_draft=False,
        )
        request_ids = list(range(batch))
        assert self.manager.add_dummy_requests(
            request_ids, token_nums=[_PREFILL + 1 + _DRAFTS] * batch
        )
        generator = torch.Generator(device=device).manual_seed(len(request_ids))
        self.metadata = {}
        self.input_ids = {}
        for kind, tokens in _TOKENS.items():
            metadata = get_attention_backend("TRTLLM").Metadata(
                seq_lens=torch.tensor([tokens] * batch, dtype=torch.int),
                request_ids=request_ids,
                max_num_requests=batch,
                num_contexts=batch if kind == "prefill" else 0,
                prompt_lens=[_PREFILL] * batch,
                max_num_tokens=batch * _PREFILL,
                kv_cache_manager=self.manager,
                kv_cache_params=KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=[0 if kind == "prefill" else _PREFILL] * batch,
                ),
                mapping=config.mapping,
            )
            metadata.mamba_metadata_cls = model.mamba_metadata_cls
            if kind == "verify":
                metadata.update_spec_dec_param(
                    batch_size=batch,
                    is_spec_decoding_enabled=True,
                    is_spec_dec_tree=False,
                    is_spec_dec_dynamic_tree=False,
                    max_draft_len=_DRAFTS,
                    max_total_draft_tokens=_DRAFTS,
                )
            if kind != "prefill":
                metadata = metadata.create_cuda_graph_metadata(batch, False, tokens - 1)
            metadata.prepare()
            self.metadata[kind] = metadata
            self.input_ids[kind] = torch.randint(
                0,
                _CUT_DOWN["vocab_size"],
                (batch * tokens,),
                generator=generator,
                device=device,
                dtype=torch.int32,
            )
        layer_caches = [
            self.manager.mamba_layer_cache(index)
            for index, layer in enumerate(model.model.layers)
            if layer.is_kda
        ]
        # The KDA verify is the fused one only with these caches, the sequential one without.
        assert all(layer_cache.kda_k_cache is not None for layer_cache in layer_caches)
        self._states = [
            state
            for layer_cache in layer_caches
            for state in vars(layer_cache).values()
            if isinstance(state, torch.Tensor)
        ]
        self._saved = None

    def restore_states(self) -> None:
        """Reset the KDA states and replay caches to the prefill's; the first call saves those."""
        if self._saved is None:
            self._saved = [state.clone() for state in self._states]
        for state, saved in zip(self._states, self._saved):
            state.copy_(saved)

    def close(self) -> None:
        torch.cuda.synchronize()
        self.manager.shutdown()


def _forward(
    model: mkl.KimiLinearForCausalLM, metadata: AttentionMetadata, input_ids: torch.Tensor
) -> dict[str, torch.Tensor]:
    """The causal LM's forward: each decoder layer's output, the final hidden state, the logits."""
    outputs = {}

    def keep_final(module, args, output) -> None:
        outputs["final"] = output.clone()

    handles = [
        layer.register_forward_hook(functools.partial(_capture_output, outputs, f"layers.{index}"))
        for index, layer in enumerate(model.model.layers)
    ]
    handles.append(model.model.register_forward_hook(keep_final))
    extra_attrs = dict(model.model_config.extra_attrs, attention_metadata=weakref.ref(metadata))
    try:
        with model_extra_attrs(extra_attrs), torch.no_grad():
            outputs["logits"] = model.forward(metadata, input_ids=input_ids)
    finally:
        for handle in handles:
            handle.remove()
    return outputs


@contextlib.contextmanager
def _delays(
    model: mkl.KimiLinearForCausalLM, launcher: _SpinningLauncher | None, delay: str
) -> Iterator[None]:
    """The spin kernels of a step's ``delay``; none without dense DWDP (no ``launcher``)."""
    with contextlib.ExitStack() as stack:
        if launcher is not None and delay == "copies":
            launcher.spin_cycles = _SPIN_CYCLES
            stack.callback(setattr, launcher, "spin_cycles", 0)
        elif launcher is not None and delay == "reads":
            stack.enter_context(_slow_windows(model, _SPIN_CYCLES))
            for layer_idx in _COVERED:
                layer = model.model.layers[layer_idx]
                readers = [layer.block_sparse_moe.shared_experts]
                if layer.is_kda:
                    readers.append(layer.linear_attn.f_b_proj)
                for module in readers:
                    hook = module.register_forward_pre_hook(lambda *_: _spin(_SPIN_CYCLES))
                    stack.callback(hook.remove)
        yield


def _run_steps(
    model: mkl.KimiLinearForCausalLM,
    cache: _GenCache,
    manager: DenseDwdpManager,
    launcher: _SpinningLauncher | None = None,
) -> list[dict[str, torch.Tensor] | None]:
    """Run ``_STEPS`` on an execution stream as ``PyExecutor`` does; None for a capture.

    Only the captures synchronize the host.  A replay runs behind the manager's replay guard, and
    ``after_forward`` follows the last step, which polls the slot watchdog.
    """
    graphs = {}
    results = []
    execution_stream = torch.cuda.Stream()
    execution_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(execution_stream):
        for step in _STEPS:
            metadata, input_ids = cache.metadata[step.kind], cache.input_ids[step.kind]
            if step.kind != "prefill" and step.mode != "capture":
                cache.restore_states()
            if step.mode == "replay":
                graph, outputs = graphs[step.kind]
                manager.before_replay()
                graph.replay()
                results.append({name: value.clone() for name, value in outputs.items()})
                continue
            with _delays(model, launcher, step.delay), with_multi_stream(step.multi_stream):
                if step.mode == "eager":
                    results.append(_forward(model, metadata, input_ids))
                    continue
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graphs[step.kind] = graph, _forward(model, metadata, input_ids)
                results.append(None)
    torch.cuda.current_stream().wait_stream(execution_stream)
    manager.after_forward()
    torch.cuda.synchronize()
    return results


def _check_gen_setup(model: mkl.KimiLinearForCausalLM, run: _DenseRun, variant: _Variant) -> None:
    """S7's setup audit on SM107; elsewhere the collectives, the covered layers and segments."""
    if _rubin():
        _check_setup(model, run, variant)
        return
    layout = run.layout
    assert run.allgathers == 5
    assert model._dense_dwdp_active and model.model._dense_dwdp_active
    assert tuple(layout.layers) == _COVERED
    assert list(layout.warnings) == []
    covered = {(layer, kind) for layer in _COVERED for kind in variant.segments}
    assert set(layout.segments) == covered
    assert all(segment.moved_bytes > 0 for segment in layout.segments.values())


def _check_gen_tactics(log: _TacticLog, reference_calls: list) -> int:
    """S7's tactic check on SM107, where the MXFP8 weight-read GEMM must run; elsewhere the calls
    and their tactics must repeat the reference run's.

    Returns:
        The number of tuned op calls checked.
    """
    if _rubin():
        _check_tactics(log, reference_calls)
    assert log.calls == reference_calls, "the GEMM calls or their tactics changed after setup"
    return len(log.calls)


#: ``(transfer, KV-cache dtype, batch)``: each transfer with each KV-cache dtype, with BF16 at the
#: batch whose projections fork onto the auxiliary stream and FP8 at the one whose verify does not.
_CASES = [
    (CopyMode.GATHER, "bf16", 4),
    (CopyMode.GATHER, "fp8", 24),
    (CopyMode.BATCHED, "bf16", 4),
    (CopyMode.BATCHED, "fp8", 24),
]


@pytest.mark.parametrize(
    ("transfer", "kv_cache", "batch"),
    _CASES,
    ids=[f"{transfer.value}-{kv_cache}_kv-batch{batch}" for transfer, kv_cache, batch in _CASES],
)
def test_k3_decode_and_verify_are_bit_exact_in_graph_mode(
    transfer: CopyMode,
    kv_cache: str,
    batch: int,
    device: torch.device,
    tactics: _TacticLog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decode and MTP verify on the slots equal them on the replicated weights, eager and replayed.

    A standalone ``DenseDwdpManager`` in graph mode is registered before the model is built, as
    ``py_executor_creator`` registers it for a generation server, and sets dense DWDP up after the
    replicated run.  At batch 24 the verify no longer forks the KDA projections.
    """
    _transfer_env(monkeypatch, transfer, device)
    variant = _gen_variant(transfer)
    mapping = _dep_mapping(variant.num_ranks, variant.rank)
    manager = DenseDwdpManager(DenseMode.STANDALONE, mapping=mapping, graph_mode=True)
    with manager:
        model = _build_gen_model(kv_cache, monkeypatch, device)
        _check_captured(model, manager)
        cache = _GenCache(model, batch, device)
        assert cache.manager.dtype == _KV_CACHE_DTYPE[kv_cache]
        try:
            references = _run_steps(model, cache, manager)
            for kind in ("decode", "verify"):
                first, *others = [
                    result
                    for step, result in zip(_STEPS, references)
                    if step.kind == kind and result is not None
                ]
                assert all(torch.isfinite(value).all() for value in first.values()), kind
                for other in others:
                    differences = _differences(first, other)
                    assert not differences, f"{kind} without dense DWDP: " + "; ".join(differences)
            reference_calls = list(tactics.calls)
            tactics.calls.clear()

            run = _set_up_dense(model, variant, device, monkeypatch, manager)
            runtime = run.runtime
            try:
                _check_gen_setup(model, run, variant)
                assert runtime.protocol is DenseProtocol.CLOSED
                assert runtime.prime_policy is PrimePolicy.WRAP
                results = _run_steps(model, cache, manager, run.launcher)
                for index, (step, expected, actual) in enumerate(zip(_STEPS, references, results)):
                    if expected is not None:
                        differences = _differences(expected, actual)
                        assert not differences, (
                            f"forward {index} ({step.kind} {step.mode}): " + "; ".join(differences)
                        )
                gemms = _check_gen_tactics(tactics, reference_calls)
                eager = sum(step.mode == "eager" for step in _STEPS)
                captures = sum(step.mode == "capture" for step in _STEPS)
                assert (runtime.eager_forwards, runtime.captures) == (eager, captures)
                assert (runtime.recoveries, runtime.demand_fetches, runtime.war_fences) == (0, 0, 0)
                # Counted at H1 of the eager forwards and the captures; replays run them too.
                checks = (eager + captures) * len(_COVERED) * len(variant.segments)
                assert (runtime.watchdog_checks, runtime.watchdog_mismatches) == (checks, 0)
                _report(
                    f"dense k3 decode {transfer.value} {kv_cache}_kv batch {batch}",
                    run,
                    f"{sum(result is not None for result in results)} forwards bit-exact "
                    f"({eager} eager, {len(_STEPS) - eager - captures} replays of {captures} "
                    f"graphs); {gemms} tuned op calls "
                    f"({'cached' if tactics.cache_loaded else 'fallback'} tactics); slot watchdog "
                    f"{runtime.watchdog_checks} checks, {runtime.watchdog_mismatches} mismatches",
                )
            finally:
                manager.cleanup()
            if _rubin():
                _check_detached(model.model.layers, variant.segments)
        finally:
            cache.close()
