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
"""The Kimi-K3 side of dense DWDP: hook placement and the load guard.

Covers the hook calls H0-H3 of ``modeling_kimi_linear`` and its load guard.  Layers are real modules
built from a tiny Kimi-Linear config.  A real forward of the attention and MoE submodules needs
attention metadata, KV-cache pools and the routed-expert kernels, so the hook tests replace exactly
those submodules with recorders; everything else in the decoder-layer and model forward runs for
real, on CPU in fp32 (the eager norm paths).

KDA layers, dense-MLP and MoE, build on CPU.  The routed experts of a MoE layer are a stand-in
without tensors, built as in ``test_kimi_k3_mlp.py``: ``create_moe`` resolves the backend for the
GPU's SM version and puts the SiTU constants of a CUTLASS backend on the current CUDA device, and
the experts never run here.  The MLA mixer (``KimiK3MLAAttention``) allocates its attention
backends' RoPE tables and scale tensors on CUDA, so the MLA cases run twice: with the real mixer,
which needs a GPU and skips without one (the parameters still stay on CPU), and on CPU with only
that mixer replaced by a stand-in.  The stand-in keeps the layer's real MLA branch and its
``KimiMLARuntime`` wrapper, which are what the roots and the hooks depend on.

The model reaches dense DWDP through the dense DWDP manager it captures when built
(``get_global_dense_dwdp_manager``), in both modes: attached to expert DWDP and standalone.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("fla")
from tensorrt_llm._torch.configs.kimi_linear import KimiLinearConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_kimi_linear as mkl
from tensorrt_llm._torch.modules import kimi_k3_mla
from tensorrt_llm._torch.modules.dwdp.dense.manager import set_global_dense_dwdp_manager
from tensorrt_llm._torch.moe.fused_moe import ConfigurableMoE, CutlassFusedMoE
from tensorrt_llm._torch.pyexecutor.dwdp import set_global_dwdp_manager
from tensorrt_llm._torch.utils import AuxStreamType
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

_HIDDEN = 256
_TOKENS = 3

# Layers of the default ``_kimi_config``: 0 is a dense-MLP KDA layer, then the KDA, MLA and KDA
# MoE layers.
_KDA_MOE_LAYER = 1
_MLA_MOE_LAYER = 2
_MOE_LAYERS = (1, 2, 3)

_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the MLA mixer's attention backends allocate their tables on CUDA",
)
# The MoE layers the root and hook tests build: KDA, MLA with its real mixer, and MLA with the
# CPU stand-in mixer.
_MOE_LAYER_KINDS = [
    pytest.param("kda", marks=pytest.mark.cpu_only),
    pytest.param("mla", marks=_requires_cuda),
    pytest.param("mla_cpu_mixer", marks=pytest.mark.cpu_only),
]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _kimi_config(
    *,
    kda_layers: tuple[int, ...] = (1, 2, 4),
    mla_layers: tuple[int, ...] = (3,),
    first_k_dense_replace: int = 1,
    full_rank_gate: bool = True,
    num_heads: int = 2,
    num_experts: int = 8,
) -> KimiLinearConfig:
    """A tiny Kimi-K3: by default a dense-MLP KDA layer 0, then KDA, MLA and KDA MoE layers.

    ``kda_layers`` and ``mla_layers`` are 1-indexed, as in the checkpoint config.  ``num_heads``
    is the head count of the KDA and the MLA attention.
    """
    return KimiLinearConfig(
        vocab_size=128,
        hidden_size=_HIDDEN,
        intermediate_size=256,
        num_hidden_layers=len(kda_layers) + len(mla_layers),
        num_attention_heads=num_heads,
        q_lora_rank=256,
        kv_lora_rank=128,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rms_norm_eps=1e-6,
        attn_res_block_size=2,
        mla_use_output_gate=True,
        max_position_embeddings=64,
        linear_attn_config={
            "kda_layers": list(kda_layers),
            "full_attn_layers": list(mla_layers),
            "num_heads": num_heads,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": full_rank_gate,
        },
        num_experts=num_experts,
        num_experts_per_token=2,
        moe_intermediate_size=128,
        num_shared_experts=1,
        routed_expert_hidden_size=128,
        latent_moe_use_norm=True,
        first_k_dense_replace=first_k_dense_replace,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )


def _kda_config(num_layers: int, *, first_k_dense_replace: int = 1) -> KimiLinearConfig:
    """``num_layers`` KDA layers: a whole model that builds on CPU."""
    return _kimi_config(
        kda_layers=tuple(range(1, num_layers + 1)),
        mla_layers=(),
        first_k_dense_replace=first_k_dense_replace,
    )


def _model_config(
    cfg: KimiLinearConfig,
    mapping: Mapping | None = None,
    quant_config_dict: dict[str, QuantConfig] | None = None,
) -> ModelConfig:
    """The deployed context-worker shape: tp1 with attention DP (fp32 router, replicated MLPs).

    Args:
        mapping: Another rank mapping, such as a DEP16 rank's.
        quant_config_dict: Per-module quantization, by checkpoint module name.
    """
    if mapping is None:
        mapping = Mapping(world_size=1, tp_size=1, rank=0, enable_attention_dp=True)
    return ModelConfig(
        pretrained_config=cfg,
        mapping=mapping,
        quant_config=QuantConfig(),
        quant_config_dict=quant_config_dict,
        moe_backend="CUTLASS",
    )


def _aux_streams() -> dict:
    # Construction only stores the auxiliary streams; nothing in these tests runs on them.
    return {kind: object() for kind in AuxStreamType}


def _build_layer(layer_idx: int, cfg: KimiLinearConfig | None = None) -> mkl.KimiLinearDecoderLayer:
    cfg = _kimi_config() if cfg is None else cfg
    return mkl.KimiLinearDecoderLayer(_model_config(cfg), cfg, layer_idx, _aux_streams())


def _build_model(monkeypatch: pytest.MonkeyPatch, cfg: KimiLinearConfig) -> mkl.KimiLinearModel:
    # The model creates its auxiliary streams eagerly; CPU stand-ins keep CUDA out of it.
    monkeypatch.setattr(torch.cuda, "Stream", object)
    return mkl.KimiLinearModel(_model_config(cfg))


class _CpuMlaMixer(nn.Module):
    """Stands in for ``KimiK3MLAAttention``, whose attention backends allocate CUDA tensors.

    Takes the mixer's keyword arguments and holds one projection, so ``KimiMLARuntime`` wraps it
    as it wraps the real mixer.
    """

    def __init__(self, *, hidden_size: int, num_heads: int, v_head_dim: int, **kwargs) -> None:
        super().__init__()
        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)


def _use_cpu_mla_mixer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build MLA layers with ``_CpuMlaMixer``; ``KimiMLARuntime`` imports the mixer per call."""
    monkeypatch.setattr(kimi_k3_mla, "KimiK3MLAAttention", _CpuMlaMixer)


class _CpuRoutedBackend(CutlassFusedMoE):
    """The CUTLASS backend the layers request, without its constructor and its tensors.

    ``CutlassFusedMoE`` is the abstract root of the CUTLASS leaves; the two methods a leaf
    implements are never called on this stand-in.
    """

    def __init__(self, expert_ids: list[int]) -> None:
        nn.Module.__init__(self)
        self.initial_local_expert_ids = expert_ids

    @classmethod
    def can_implement(cls, p, d):
        raise NotImplementedError("CPU stand-in")

    def _get_quant_method(self):
        raise NotImplementedError("CPU stand-in")


class _CpuRoutedExperts(ConfigurableMoE):
    """Stands in for the routed experts ``create_moe`` builds.

    Holds what ``KimiK3MoERuntime`` reads at construction: a backend with this rank's contiguous
    slice of the experts under expert parallelism (all of them at tp1), no load balancer and no
    all-reduce.
    """

    def __init__(self, *, num_experts: int, model_config: ModelConfig, **kwargs) -> None:
        nn.Module.__init__(self)
        mapping = model_config.mapping
        per_rank = num_experts // mapping.moe_ep_size
        first = mapping.moe_ep_rank * per_rank
        self.backend = _CpuRoutedBackend(list(range(first, first + per_rank)))
        self.comm = None
        self.layer_load_balancer = None
        self.all_reduce = None


@pytest.fixture
def cpu_routed_experts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build the routed experts as ``_CpuRoutedExperts``, so MoE layers build on CPU."""
    monkeypatch.setattr(mkl, "create_moe", lambda **kwargs: _CpuRoutedExperts(**kwargs))


@pytest.fixture(params=_MOE_LAYER_KINDS)
def moe_layer(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, cpu_routed_experts: None
) -> mkl.KimiLinearDecoderLayer:
    """A MoE layer of the default ``_kimi_config``, of each kind in ``_MOE_LAYER_KINDS``."""
    kind = request.param
    if kind == "mla_cpu_mixer":
        _use_cpu_mla_mixer(monkeypatch)
    layer = _build_layer(_KDA_MOE_LAYER if kind == "kda" else _MLA_MOE_LAYER)
    assert layer.is_moe
    assert layer.is_kda == (kind == "kda")
    if not layer.is_kda:
        assert isinstance(layer.self_attn, mkl.KimiMLARuntime)
        assert isinstance(layer.self_attn.mixer, _CpuMlaMixer) == (kind == "mla_cpu_mixer")
    return layer


@pytest.fixture(autouse=True)
def _no_global_managers():
    """Layers capture the global dense manager when built; every test starts and ends without one.

    The DWDP manager's accessor is reset too: a leftover registration of either fails the next
    ``DwdpManager`` of this process.
    """
    set_global_dwdp_manager(None)
    set_global_dense_dwdp_manager(None)
    yield
    set_global_dwdp_manager(None)
    set_global_dense_dwdp_manager(None)


# ---------------------------------------------------------------------------
# Recorders and fakes
# ---------------------------------------------------------------------------


class _Recorder(nn.Module):
    """Stands in for an attention, MoE or MLP submodule: logs the call, returns a scaled input."""

    def __init__(self, log: list, name: str, layer_idx: int) -> None:
        super().__init__()
        self._log = log
        self._name = name
        self._layer_idx = layer_idx
        self.fail = False

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self._log.append((self._name, self._layer_idx))
        if self.fail:
            raise RuntimeError(f"{self._name} of layer {self._layer_idx} failed")
        return hidden_states * 0.5


class _Embedding(nn.Module):
    """Stands in for the token embedding: logs the call, returns fp32 embeddings."""

    def __init__(self, log: list) -> None:
        super().__init__()
        self._log = log

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self._log.append(("embed",))
        return torch.randn(input_ids.shape[0], _HIDDEN)


class _RecordingDense:
    """The ``DenseDWDP`` calls the model makes, recorded; it covers the layers in ``covered``.

    Any other runtime call raises ``AttributeError``.
    """

    def __init__(self, log: list, covered: tuple[int, ...]) -> None:
        self._log = log
        self._covered = frozenset(covered)

    def covers(self, layer: int) -> bool:
        self._log.append(("covers", layer))
        return layer in self._covered

    def wait_layer(self, layer: int) -> None:
        self._log.append(("wait_layer", layer))

    def attention_done(self, layer: int) -> None:
        self._log.append(("attention_done", layer))

    def layer_done(self, layer: int) -> None:
        self._log.append(("layer_done", layer))


class _RecordingManager:
    """``DenseDwdpManager.dense`` and ``DenseDwdpManager.ensure_primed``, recorded."""

    def __init__(self, log: list, dense: object | None) -> None:
        self._log = log
        self._dense = dense

    @property
    def dense(self) -> object | None:
        return self._dense

    def ensure_primed(self) -> None:
        self._log.append(("ensure_primed",))


def _install_recorders(layer: mkl.KimiLinearDecoderLayer, log: list) -> _Recorder:
    """Replace the layer's attention and MoE (or MLP) submodules; log its norms' calls too.

    Returns:
        The recorder standing in for the attention.
    """
    attention = _Recorder(log, "attention", layer.layer_idx)
    setattr(layer, "linear_attn" if layer.is_kda else "self_attn", attention)
    if layer.is_moe:
        layer.block_sparse_moe = _Recorder(log, "moe", layer.layer_idx)
    else:
        layer.mlp = _Recorder(log, "mlp", layer.layer_idx)
    for norm in ("input_layernorm", "post_attention_layernorm"):
        getattr(layer, norm).register_forward_pre_hook(
            lambda module, args, name=norm: log.append((name, layer.layer_idx))
        )
    return attention


def _attn_metadata() -> SimpleNamespace:
    # The model forward checks the batch against ``num_tokens``; the recorders read nothing.
    return SimpleNamespace(num_tokens=_TOKENS)


def _run_layer(layer: mkl.KimiLinearDecoderLayer) -> None:
    # fp32 activations keep the residual mixing and the norms on their eager CPU paths.
    hidden_states = torch.randn(_TOKENS, _HIDDEN)
    block_residual = torch.zeros(2, _TOKENS, _HIDDEN)
    layer(hidden_states, block_residual, 0, _attn_metadata())


def _layer_reads(layer_idx: int, ffn: str) -> list[tuple]:
    """The submodule and norm calls of one decoder-layer forward, in program order."""
    return [
        ("input_layernorm", layer_idx),
        ("attention", layer_idx),
        ("post_attention_layernorm", layer_idx),
        (ffn, layer_idx),
    ]


def _hooked_layer_reads(layer_idx: int, ffn: str) -> list[tuple]:
    """``_layer_reads`` of a covered layer: H1 at entry, H2 after attention, H3 after the MoE."""
    return [
        ("covers", layer_idx),
        ("wait_layer", layer_idx),
        ("input_layernorm", layer_idx),
        ("attention", layer_idx),
        ("attention_done", layer_idx),
        ("post_attention_layernorm", layer_idx),
        (ffn, layer_idx),
        ("layer_done", layer_idx),
    ]


def test_hooks_bracket_the_attention_and_moe_reads(moe_layer):
    log = []
    _install_recorders(moe_layer, log)
    moe_layer._dense_dwdp = _RecordingManager(log, _RecordingDense(log, covered=_MOE_LAYERS))
    _run_layer(moe_layer)
    assert log == _hooked_layer_reads(moe_layer.layer_idx, "moe")


# ---------------------------------------------------------------------------
# H0: priming in the model forward
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
@pytest.mark.parametrize("state", ["dense", "no_dense", "no_manager"])
def test_model_primes_before_the_embedding_only_with_dense(monkeypatch, state):
    log = []
    model = _build_model(monkeypatch, _kda_config(1))
    _install_recorders(model.layers[0], log)
    model.embed_tokens = _Embedding(log)
    manager = {
        "dense": _RecordingManager(log, _RecordingDense(log, covered=_MOE_LAYERS)),
        "no_dense": _RecordingManager(log, None),
        "no_manager": None,
    }[state]
    model._dense_dwdp = manager
    model.layers[0]._dense_dwdp = manager

    output = model(_attn_metadata(), input_ids=torch.zeros(_TOKENS, dtype=torch.int32))

    assert output.shape == (_TOKENS, _HIDDEN)
    if state == "dense":
        assert log == [("ensure_primed",), ("embed",), ("covers", 0)] + _layer_reads(0, "mlp")
    else:
        assert log == [("embed",)] + _layer_reads(0, "mlp")

    # The input check runs first: a rejected call does not prime.
    log.clear()
    with pytest.raises(ValueError, match="exactly one of input_ids or inputs_embeds"):
        model(_attn_metadata())
    assert log == []


# ---------------------------------------------------------------------------
# The load guard
# ---------------------------------------------------------------------------


class _LoadReached(Exception):
    """Raised by the stubbed loader once ``load_weights`` got past its guard."""


def _causal_lm(active: bool | None) -> mkl.KimiLinearForCausalLM:
    # The guard reads only ``self.model``; the first loader step is stubbed to report reaching it.
    model = mkl.KimiLinearForCausalLM.__new__(mkl.KimiLinearForCausalLM)
    nn.Module.__init__(model)
    model.model = nn.Module()
    if active is not None:
        model.model._dense_dwdp_active = active

    def trunk_parameters():
        raise _LoadReached

    model._trunk_parameters = trunk_parameters
    return model


@pytest.mark.cpu_only
def test_load_weights_refuses_while_dense_is_active():
    with pytest.raises(RuntimeError, match="dense DWDP is active"):
        _causal_lm(active=True).load_weights({})
    for active in (False, None):
        with pytest.raises(_LoadReached):
            _causal_lm(active=active).load_weights({})
