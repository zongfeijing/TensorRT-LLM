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
"""CPU tests of the dense DWDP collector.

The toy decoder layers below imitate the Kimi-K3 tensors the collector has to handle: the fused
``qkvg_proj.weight`` buffer with ``q/k/v/g_proj.weight`` row-view buffers, the plain
``_bfa_proj_weight`` attribute with ``f_a_proj`` / ``b_proj`` Parameter row views and pad rows no
view covers, the ``_mtp_conv_weights`` tuple, ``_A_log_f32`` sharing ``A_log``'s storage, tensors
under the threshold, excluded scratch names (one of them aliasing an anchor), an MLA layer and the
MoE-dense roots.  Aliases that are not plain row views ride along: ``_gate_b_t`` is a transposed
view of ``f_b_proj``'s storage (the low-rank gate's layout), and ``_bfa_views`` holds a stride-0
view and an empty view of the ``_bfa`` storage.  Weights hold small integers so every toy forward
is exact in fp32 and can be compared bit for bit (``test_dense_repoint.py`` reuses the toy).
"""

import weakref

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.modules.dwdp.dense.config import DenseConfig
from tensorrt_llm._torch.modules.dwdp.dense.layout import collect_dense_layout
from tensorrt_llm._torch.modules.dwdp.dense.types import BindingKind, DenseLayoutError, SegmentKind

pytestmark = pytest.mark.cpu_only

ATTN = SegmentKind.ATTN
MOE = SegmentKind.MOE

#: The model tables, as ``modeling_kimi_linear`` defines them.
ATTR_ALLOW = frozenset(
    {
        "_bfa_proj_weight",
        "_qkvg_proj_weight",
        "_gate_b_t",
        "_packed_conv_weight",
        "_mtp_conv_weights",
        "_w_q_t",
        "_w_k_t",
        "_w_v_t",
        "_A_log_f32",
        "_dt_bias_f32",
        "_onorm_w_f32",
    }
)
EXCLUDE = frozenset(
    {
        "_o_dense",
        "_fused_q_placeholder",
        "softmax_stats_tensor",
        "temp_softmax_stats_tensor",
        "e_score_correction_bias",
    }
)

#: Toy threshold: the toy weights are 8-64 KiB, the small tensors at most 2 KiB.
TOY_MIN_BYTES = 8192
NUM_RANKS = 4
HIDDEN = 256
ROWS = 64
F_ROWS = 16
B_ROWS = 2
#: Decoder layer kinds: layer 0 stays replicated, layer 3 is an MLA layer.
LAYER_KINDS = ("dense", "kda", "kda", "mla", "kda", "kda")
COVERED = (1, 2, 3, 4, 5)

FP8 = torch.float8_e4m3fn
BF16 = torch.bfloat16


# ---------------------------------------------------------------------------
# Toy Kimi-K3 decoder layers
# ---------------------------------------------------------------------------


def _ints(generator, shape, dtype):
    """Small integers in [-3, 3], exactly representable in every moved dtype."""
    return torch.randint(-3, 4, shape, generator=generator).to(torch.float32).to(dtype)


def _bytes(generator, shape):
    return torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8)


def _squash(x):
    """Fold back into [-3, 3]; exact on the integer-valued fp32 tensors of the toy."""
    return torch.remainder(x, 7.0) - 3.0


def _project(x, weight, scale=None):
    out = x @ weight.float().t()
    if scale is not None:
        out = out + scale.float().sum()
    return _squash(out)


def _parameter(tensor):
    return nn.Parameter(tensor, requires_grad=False)


class ToyFp8Linear(nn.Module):
    """Stands in for ``_Fp8BlockScaleWeightReadLinear``: weight and scale buffers."""

    def __init__(self, weight, scale):
        super().__init__()
        self.out_features = weight.shape[0]
        self._weights_transformed = True
        self._fused_projection = None
        self.register_buffer("weight", weight, persistent=False)
        self.register_buffer("weight_scale", scale, persistent=False)

    def forward(self, x):
        return _project(x, self.weight, self.weight_scale)


class ToyNorm(nn.Module):
    def __init__(self, generator, size, dtype=BF16):
        super().__init__()
        self.weight = _parameter(_ints(generator, (size,), dtype))


class ToyLinear(nn.Linear):
    def __init__(self, generator, in_features, out_features):
        super().__init__(in_features, out_features, bias=False, dtype=BF16)
        self.weight.requires_grad_(False)
        self.weight.data.copy_(_ints(generator, (out_features, in_features), BF16))


class ToyKDA(nn.Module):
    """KDA mixer after load: fused FP8 qkvg, fused BF16 [f_a | b], decode constants."""

    def __init__(self, generator):
        super().__init__()
        g = generator
        self.A_log = _parameter(_ints(g, (16,), torch.float32))
        self.dt_bias = _parameter(_ints(g, (64,), torch.float32))
        for name in ("q_proj", "k_proj", "v_proj", "g_proj"):
            setattr(self, name, ToyFp8Linear(_ints(g, (ROWS, HIDDEN), FP8), _bytes(g, (ROWS, 128))))
        self.o_proj = ToyFp8Linear(_ints(g, (HIDDEN, 4 * ROWS), FP8), _bytes(g, (HIDDEN, 32)))
        self.f_a_proj = ToyLinear(g, HIDDEN, F_ROWS)
        self.b_proj = ToyLinear(g, HIDDEN, B_ROWS)
        self.f_b_proj = ToyLinear(g, F_ROWS, HIDDEN)
        self.q_conv1d = ToyNorm(g, 4 * HIDDEN)
        self.o_norm = ToyNorm(g, HIDDEN)
        self._finalize(g)

    def _finalize(self, g):
        """Mirror ``finalize_decode_weights`` and the FP8 fusion of the K3 loader."""
        projections = [self.q_proj, self.k_proj, self.v_proj, self.g_proj]
        fused = torch.cat([module.weight for module in projections])
        self.qkvg_proj = ToyFp8Linear(fused, _bytes(g, (4 * ROWS, 64)))
        for index, module in enumerate(projections):
            module._fused_projection = (weakref.ref(self.qkvg_proj), index * ROWS)
            module.weight = fused[index * ROWS : (index + 1) * ROWS]
        bfa = torch.cat(
            [self.f_a_proj.weight.data, self.b_proj.weight.data, torch.zeros(6, HIDDEN, dtype=BF16)]
        )
        self.f_a_proj.weight.data = bfa[:F_ROWS]
        self.b_proj.weight.data = bfa[F_ROWS : F_ROWS + B_ROWS]
        self._bfa_proj_weight = bfa
        # Names in neither model table that alias the _bfa storage: a stride-0 view and an empty
        # view past the pad rows.
        self._bfa_views = [bfa[:1].expand(4, -1), bfa[bfa.shape[0] :]]
        self._qkvg_proj_weight = None
        # The low-rank gate's [f_b; g_b] bmm operand: a transposed view of f_b_proj's storage.
        self._gate_b_t = self.f_b_proj.weight.data.view(2, HIDDEN // 2, F_ROWS).transpose(1, 2)
        self._mtp_conv_weights = tuple(_ints(g, (4, 1024), BF16) for _ in range(3))
        self._packed_conv_weight = _ints(g, (3, HIDDEN), BF16)
        self._w_q_t = _ints(g, (4, 128), BF16)
        self._A_log_f32 = self.A_log.detach().float().contiguous()
        self._dt_bias_f32 = self.dt_bias.detach().float().contiguous()
        self._o_dense = self.o_proj.weight[64:192]
        self._projection_fork_event = object()

    def forward(self, x):
        heads = torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x), self.g_proj(x)], -1)
        heads = _squash(heads + self.qkvg_proj(x))
        f_a = _project(x, self.f_a_proj.weight)
        beta = _project(x, self.b_proj.weight)
        bfa = _project(x, self._bfa_proj_weight)
        f_b = _project(f_a, self.f_b_proj.weight)
        gates = torch.bmm(f_a.expand(2, -1, -1), self._gate_b_t.float())
        gates = _squash(gates.transpose(0, 1).reshape(f_a.shape[0], -1))
        views = sum(_project(x, view).sum(-1, keepdim=True) for view in self._bfa_views)
        constants = (
            sum(weight.float().sum() for weight in self._mtp_conv_weights)
            + self._A_log_f32.sum()
            + self._dt_bias_f32.sum()
            + self._packed_conv_weight.float().sum()
            + self._o_dense.float().sum()
        )
        mixed = heads + f_b + gates + bfa.sum(-1, keepdim=True) + beta.sum(-1, keepdim=True)
        return self.o_proj(_squash(mixed + views + constants + self.o_norm.weight.float()))


class ToyAttentionBackend:
    """Not an ``nn.Module``: the walk never enters it (like MLA's ``mha`` / ``mqa``)."""

    def __init__(self, generator):
        self.workspace = _ints(generator, (4096,), BF16)


class ToyMLA(nn.Module):
    """MLA attention after load, with the BF16 absorb copies and excluded scratch tensors."""

    def __init__(self, generator):
        super().__init__()
        g = generator
        self.kv_a_proj_with_mqa = ToyFp8Linear(_ints(g, (128, HIDDEN), FP8), _bytes(g, (128, 64)))
        self.q_b_proj = ToyFp8Linear(_ints(g, (HIDDEN, 128), FP8), _bytes(g, (HIDDEN, 32)))
        self.kv_b_proj = ToyFp8Linear(_ints(g, (HIDDEN, 64), FP8), _bytes(g, (HIDDEN, 32)))
        self.o_proj = ToyFp8Linear(_ints(g, (HIDDEN, HIDDEN), FP8), _bytes(g, (HIDDEN, 32)))
        self.k_b_proj_trans = _parameter(_ints(g, (8, 64, 64), FP8))
        self.k_b_proj_trans_dequant = _parameter(_ints(g, (8, 64, 64), BF16))
        self.v_b_proj_scale = _parameter(self.kv_b_proj.weight_scale[:128])
        self._fused_q_placeholder = _ints(g, (64, HIDDEN), BF16)
        self.softmax_stats_tensor = torch.zeros(4096, dtype=torch.float32)
        self.mha = ToyAttentionBackend(g)

    def forward(self, x):
        latent = self.kv_a_proj_with_mqa(x)
        query = self.q_b_proj(latent)
        key_value = self.kv_b_proj(latent[:, :64])
        absorb = (
            self.k_b_proj_trans.float().sum()
            + self.k_b_proj_trans_dequant.float().sum()
            + self.v_b_proj_scale.float().sum()
        )
        return self.o_proj(_squash(query + key_value + absorb))


class ToyGate(nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.weight = _parameter(_ints(generator, (32, HIDDEN), torch.float32))
        self.e_score_correction_bias = _parameter(_ints(generator, (32,), torch.float32))


class ToySharedMLP(nn.Module):
    def __init__(self, generator):
        super().__init__()
        g = generator
        self.gate_up_proj = ToyFp8Linear(_ints(g, (HIDDEN, HIDDEN), FP8), _bytes(g, (HIDDEN, 32)))
        self.down_proj = ToyFp8Linear(_ints(g, (HIDDEN, 128), FP8), _bytes(g, (HIDDEN, 32)))

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(_squash(gate * up))


class ToyRoutedExperts(nn.Module):
    """Managed by expert DWDP; never a dense root."""

    def __init__(self, generator):
        super().__init__()
        self.weight = _parameter(_ints(generator, (4, HIDDEN, HIDDEN), BF16))


class ToyMoE(nn.Module):
    def __init__(self, generator):
        super().__init__()
        g = generator
        self.gate = ToyGate(g)
        self.shared_experts = ToySharedMLP(g)
        self.routed_expert_down_proj = ToyFp8Linear(
            _ints(g, (128, HIDDEN), FP8), _bytes(g, (128, 16))
        )
        self.routed_expert_up_proj = ToyFp8Linear(
            _ints(g, (HIDDEN, 128), FP8), _bytes(g, (HIDDEN, 8))
        )
        self.routed_expert_norm = ToyNorm(g, 128)
        self.routed_experts = ToyRoutedExperts(g)

    def forward(self, x):
        logits = _squash(x @ self.gate.weight.t() + self.gate.e_score_correction_bias)
        latent = self.routed_expert_down_proj(x)
        latent = _squash(
            latent * self.routed_expert_norm.weight.float() + logits.sum(-1, keepdim=True)
        )
        expert = _squash(latent + self.routed_experts.weight[0, 0, :128].float())
        return _squash(self.routed_expert_up_proj(expert) + self.shared_experts(x))


class ToyDecoderLayer(nn.Module):
    """Decoder layer with the ``dense_dwdp_roots`` model seam (``types.DenseRootsProvider``)."""

    def __init__(self, layer_idx, kind, generator):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_moe = kind != "dense"
        self.is_kda = kind != "mla"
        self.input_layernorm = ToyNorm(generator, HIDDEN)
        if self.is_kda:
            self.linear_attn = ToyKDA(generator)
        else:
            self.self_attn = ToyMLA(generator)
        if self.is_moe:
            self.block_sparse_moe = ToyMoE(generator)
        else:
            self.mlp = ToySharedMLP(generator)

    def dense_dwdp_roots(self, segment):
        if not self.is_moe:
            return []
        if segment == "attn":
            if self.is_kda:
                return [("linear_attn", self.linear_attn)]
            return [("self_attn", self.self_attn)]
        moe = self.block_sparse_moe
        return [
            (f"block_sparse_moe.{name}", getattr(moe, name))
            for name in (
                "gate",
                "shared_experts",
                "routed_expert_down_proj",
                "routed_expert_up_proj",
                "routed_expert_norm",
            )
        ]

    def forward(self, x):
        hidden = _squash(x + self.input_layernorm.weight.float())
        attention = self.linear_attn(hidden) if self.is_kda else self.self_attn(hidden)
        hidden = _squash(x + attention)
        mlp = self.block_sparse_moe(hidden) if self.is_moe else self.mlp(hidden)
        return _squash(hidden + mlp)


class ToyModel(nn.Module):
    def __init__(self, seed=0, kinds=LAYER_KINDS):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.layers = nn.ModuleList(
            ToyDecoderLayer(index, kind, generator) for index, kind in enumerate(kinds)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def make_config(**overrides):
    return DenseConfig(enabled=True, **{"min_bytes": TOY_MIN_BYTES, **overrides})


def collect(model, config=None, *, num_ranks=NUM_RANKS):
    return collect_dense_layout(
        list(enumerate(model.layers)),
        config or make_config(),
        num_ranks=num_ranks,
        device=torch.device("cpu"),
        attr_allow=ATTR_ALLOW,
        exclude=EXCLUDE,
    )


def anchor_paths(layout, layer, segment):
    return [
        [binding.qualified_name for binding in anchor.bindings]
        for anchor in layout.segment(layer, segment).anchors
    ]


def storage_ptr(tensor):
    return tensor.untyped_storage().data_ptr()


#: Anchors of a toy KDA layer in walk order: the mixer's own attributes (sorted by name), then
#: the children in registration order (``qkvg_proj`` last, as the K3 loader adds it).  Each entry
#: lists the anchor's bindings in walk order.
KDA_ANCHORS = [
    [
        "linear_attn._bfa_proj_weight",
        "linear_attn._bfa_views[0]",
        "linear_attn._bfa_views[1]",
        "linear_attn.f_a_proj.weight",
        "linear_attn.b_proj.weight",
    ],
    ["linear_attn._gate_b_t", "linear_attn.f_b_proj.weight"],
    ["linear_attn._mtp_conv_weights[0]"],
    ["linear_attn._mtp_conv_weights[1]"],
    ["linear_attn._mtp_conv_weights[2]"],
    [
        "linear_attn.q_proj.weight",
        "linear_attn.k_proj.weight",
        "linear_attn.v_proj.weight",
        "linear_attn.g_proj.weight",
        "linear_attn.qkvg_proj.weight",
    ],
    ["linear_attn.q_proj.weight_scale"],
    ["linear_attn.k_proj.weight_scale"],
    ["linear_attn.v_proj.weight_scale"],
    ["linear_attn.g_proj.weight_scale"],
    ["linear_attn._o_dense", "linear_attn.o_proj.weight"],
    ["linear_attn.o_proj.weight_scale"],
    ["linear_attn.qkvg_proj.weight_scale"],
]

MLA_ANCHORS = [
    ["self_attn.k_b_proj_trans"],
    ["self_attn.k_b_proj_trans_dequant"],
    ["self_attn.v_b_proj_scale", "self_attn.kv_b_proj.weight_scale"],
    ["self_attn.kv_a_proj_with_mqa.weight"],
    ["self_attn.kv_a_proj_with_mqa.weight_scale"],
    ["self_attn.q_b_proj.weight"],
    ["self_attn.q_b_proj.weight_scale"],
    ["self_attn.kv_b_proj.weight"],
    ["self_attn.o_proj.weight"],
    ["self_attn.o_proj.weight_scale"],
]

MOE_ANCHORS = [
    ["block_sparse_moe.gate.weight"],
    ["block_sparse_moe.shared_experts.gate_up_proj.weight"],
    ["block_sparse_moe.shared_experts.gate_up_proj.weight_scale"],
    ["block_sparse_moe.shared_experts.down_proj.weight"],
    ["block_sparse_moe.shared_experts.down_proj.weight_scale"],
    ["block_sparse_moe.routed_expert_down_proj.weight"],
    ["block_sparse_moe.routed_expert_up_proj.weight"],
]


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def test_collects_k3_like_anchors_and_bindings():
    model = ToyModel()
    layout = collect(model)

    assert layout.layers == COVERED
    assert layout.warnings == []
    assert list(layout.segments) == [(layer, kind) for layer in COVERED for kind in (ATTN, MOE)]
    for layer in COVERED:
        expected = MLA_ANCHORS if LAYER_KINDS[layer] == "mla" else KDA_ANCHORS
        assert anchor_paths(layout, layer, ATTN) == expected
        assert anchor_paths(layout, layer, MOE) == MOE_ANCHORS

    kda = model.layers[1].linear_attn
    anchors = {
        anchor.bindings[0].qualified_name: anchor for anchor in layout.segment(1, ATTN).anchors
    }
    qkvg = anchors["linear_attn.q_proj.weight"]
    assert qkvg.nbytes == 4 * ROWS * HIDDEN
    assert [binding.byte_offset for binding in qkvg.bindings] == [0, 16384, 32768, 49152, 0]
    assert [binding.kind for binding in qkvg.bindings] == [BindingKind.BUFFER] * 5
    # The whole _bfa storage moves, including the pad rows no Parameter view covers.
    bfa = anchors["linear_attn._bfa_proj_weight"]
    assert bfa.nbytes == 24 * HIDDEN * 2
    assert [(binding.kind, binding.byte_offset) for binding in bfa.bindings] == [
        (BindingKind.ATTRIBUTE, 0),
        (BindingKind.SEQUENCE_ITEM, 0),
        (BindingKind.SEQUENCE_ITEM, bfa.nbytes),
        (BindingKind.PARAMETER, 0),
        (BindingKind.PARAMETER, F_ROWS * HIDDEN * 2),
    ]
    assert bfa.bindings[3].original.data_ptr() == kda.f_a_proj.weight.data_ptr()
    # Bindings that are not row views keep their geometry: stride 0, empty, transposed.
    stride_0, empty = bfa.bindings[1:3]
    assert (stride_0.size, stride_0.stride) == ((4, HIDDEN), (0, 1))
    assert stride_0.span_bytes == 2 * HIDDEN
    assert (empty.size, empty.span_bytes) == ((0, HIDDEN), 0)
    gate_b_t = anchors["linear_attn._gate_b_t"].bindings[0]
    assert (gate_b_t.size, gate_b_t.stride) == (
        (2, F_ROWS, HIDDEN // 2),
        (HIDDEN // 2 * F_ROWS, 1, F_ROWS),
    )
    assert not gate_b_t.original.is_contiguous()
    # The excluded scratch name that aliases o_proj is a binding of o_proj's anchor.
    o_dense = anchors["linear_attn._o_dense"].bindings[0]
    assert (o_dense.size, o_dense.stride, o_dense.byte_offset) == ((128, 256), (256, 1), 16384)
    mtp = anchors["linear_attn._mtp_conv_weights[1]"].bindings[0]
    assert (mtp.kind, mtp.name, mtp.index) == (BindingKind.SEQUENCE_ITEM, "_mtp_conv_weights", 1)
    # Sources are flat byte views of the whole original storage.
    for anchor in layout.anchors():
        assert anchor.source.dtype == torch.uint8 and anchor.source.numel() == anchor.nbytes
        assert storage_ptr(anchor.source) == storage_ptr(anchor.bindings[0].original)
        for binding in anchor.bindings:
            assert binding.owner_ref() is not None and binding.checksum is None
    # Checksums are left to compute_checksums.
    assert all(segment.checksum is None for segment in layout.segments.values())


def test_storage_shared_across_layers_stays():
    model = ToyModel()
    model.layers[4].block_sparse_moe.gate.weight = model.layers[1].block_sparse_moe.gate.weight

    layout = collect(model)
    assert len(layout.warnings) == 1
    assert "layers.1/moe, layers.4/moe" in layout.warnings[0]
    for layer in (1, 4):
        assert anchor_paths(layout, layer, MOE) == MOE_ANCHORS[1:]
    assert anchor_paths(layout, 2, MOE) == MOE_ANCHORS

    with pytest.raises(DenseLayoutError, match="layers.1.block_sparse_moe.gate.weight"):
        collect(model, make_config(strict=True))


# ---------------------------------------------------------------------------
# Errors regardless of STRICT
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strict", [False, True])
def test_untransformed_weights_raise(strict):
    model = ToyModel()
    model.layers[4].linear_attn.o_proj._weights_transformed = False
    with pytest.raises(DenseLayoutError, match="_weights_transformed=False"):
        collect(model, make_config(strict=strict))
