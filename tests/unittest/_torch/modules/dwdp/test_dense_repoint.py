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
"""CPU tests of repointing dense bindings into staging slots.

The K3-like toy decoder layers come from ``test_dense_layout.py``.  CPU slot rings stand in for
the staging slots.
"""

import weakref

import pytest
import torch
from test_dense_layout import ToyModel, collect, storage_ptr

from tensorrt_llm._torch.modules.dwdp.dense import repoint as dense_repoint
from tensorrt_llm._torch.modules.dwdp.dense.layout import compute_checksums
from tensorrt_llm._torch.modules.dwdp.dense.repoint import BindingState, DenseBindings
from tensorrt_llm._torch.modules.dwdp.dense.types import (
    ROW_ALIGN,
    DenseRepointError,
    SegmentKind,
    SlotBuffer,
)

pytestmark = pytest.mark.cpu_only

ATTN = SegmentKind.ATTN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


#: Offset of the misaligned slot allocations from a ``ROW_ALIGN`` boundary.
MISALIGNMENT = 192


def misaligned_ring(spec):
    """Slots whose allocations start ``MISALIGNMENT`` bytes past a 4 KiB boundary.

    CPU allocations of slot size tend to be page aligned, which would leave ``pad == 0``; these
    slots need ``pad = ROW_ALIGN - MISALIGNMENT``, so every view must honour it.
    """
    ring = []
    for index in range(spec.depth):
        raw = bytearray(spec.alloc_bytes + ROW_ALIGN)
        shift = (MISALIGNMENT - torch.frombuffer(raw, dtype=torch.uint8).data_ptr()) % ROW_ALIGN
        storage = torch.frombuffer(raw, dtype=torch.uint8, offset=shift, count=spec.alloc_bytes)
        assert storage.storage_offset() == 0
        assert storage.data_ptr() % ROW_ALIGN == MISALIGNMENT
        ring.append(SlotBuffer(spec.segment, index, storage, ROW_ALIGN - MISALIGNMENT))
    return tuple(ring)


def setup_bindings(model, config=None, *, allocate=misaligned_ring):
    """P1 (collect, checksums) and the slot rings, as setup prepares them before P5."""
    layout = collect(model, config)
    compute_checksums(layout)
    slots = {kind: allocate(layout.slot_specs[kind]) for kind in layout.segment_kinds}
    return layout, slots, DenseBindings(layout, slots)


def slot_of(layout, slots, layer, kind):
    return slots[kind][layout.slot_index(kind, layer)]


class _CaptureLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(message)


@pytest.mark.parametrize("strict, describe", [(False, False), (False, True), (True, True)])
def test_hidden_holder_is_reported_with_its_path(monkeypatch, strict, describe):
    capture = _CaptureLogger()
    monkeypatch.setattr(dense_repoint, "logger", capture)
    model = ToyModel()
    kda = model.layers[2].linear_attn
    hidden = {"stash": kda.f_b_proj.weight.data}
    mla = model.layers[3].self_attn
    mla.mha.cached_absorb = mla.k_b_proj_trans.data  # a non-module object the walk never enters
    layout, slots, bindings = setup_bindings(model)
    survivor = weakref.ref(hidden["stash"].untyped_storage())
    bindings.repoint()

    # An anchor is named after its first binding; the report lists all of them.
    expected = ["layers.2.linear_attn._gate_b_t", "layers.3.self_attn.k_b_proj_trans"]
    if strict:
        with pytest.raises(DenseRepointError, match="still alive after commit") as error:
            bindings.commit(strict=True, describe_referrers=describe)
        assert all(path in str(error.value) for path in expected)
    else:
        assert bindings.commit(strict=False, describe_referrers=describe) == expected
    assert bindings.state is BindingState.COMMITTED
    assert len(capture.warnings) == 2
    for path, warning in zip(expected, capture.warnings, strict=True):
        assert f"storage of {path} " in warning
    assert "bindings (linear_attn._gate_b_t, linear_attn.f_b_proj.weight)" in capture.warnings[0]
    assert ("dict key 'stash'" in capture.warnings[0]) is describe
    assert ("holders:" in capture.warnings[1]) is describe
    # Leaking costs memory only: the module reads the slot, the holder keeps an old copy.
    assert survivor() is not None
    assert storage_ptr(kda.f_b_proj.weight) == storage_ptr(slot_of(layout, slots, 2, ATTN).storage)
    assert storage_ptr(hidden["stash"]) != storage_ptr(kda.f_b_proj.weight)
