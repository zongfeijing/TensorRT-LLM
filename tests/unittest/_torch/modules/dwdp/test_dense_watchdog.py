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
"""CPU tests of the slot checksum watchdog of ``TRTLLM_DWDP_DENSE_DEBUG=2``.

* ``SlotWatchdog`` accepts exactly the slots a
  correct gather leaves.
"""

import pytest
import torch
from test_dense_layout import ToyModel, collect

from tensorrt_llm._torch.modules.dwdp.dense.layout import compute_checksums, write_blob
from tensorrt_llm._torch.modules.dwdp.dense.types import (
    ROW_ALIGN,
    DenseLayoutError,
    SegmentKind,
    allocate_slot_ring,
)
from tensorrt_llm._torch.modules.dwdp.dense.watchdog import SlotWatchdog

pytestmark = pytest.mark.cpu_only

ATTN = SegmentKind.ATTN
MOE = SegmentKind.MOE
#: Blob bytes per ``(layer, segment)``: whole words, partial words and an empty blob.
BLOB_BYTES = {
    (1, ATTN): 5000,
    (1, MOE): 0,
    (2, ATTN): 4096,
    (2, MOE): 3,
    (3, ATTN): 2 * ROW_ALIGN - 1,
    (3, MOE): 1,
}


def blob(layer: int, kind: SegmentKind) -> torch.Tensor:
    """The bytes a correct gather of ``(layer, kind)`` leaves in its slot."""
    first = 5 * layer + (kind is MOE) + 1
    return (torch.arange(BLOB_BYTES[(layer, kind)]) * 3 + first).remainder(253).to(torch.uint8)


def test_built_from_a_layout_it_accepts_exactly_the_gathered_blobs():
    """The P1 checksums are what a gather leaves in a slot: the blob, zero-padded."""
    layout = collect(ToyModel())
    with pytest.raises(DenseLayoutError, match="not computed"):
        SlotWatchdog.from_layout(layout, "cpu")
    compute_checksums(layout)
    watchdog = SlotWatchdog.from_layout(layout, "cpu")
    assert watchdog.layers == layout.layers and watchdog.segments == layout.segment_kinds
    rings = {
        kind: allocate_slot_ring(layout.slot_specs[kind], "cpu") for kind in layout.segment_kinds
    }
    for layer in layout.layers:
        for segment in layout.layer_segments(layer):
            slot = rings[segment.segment][layout.slot_index(segment.segment, layer)]
            write_blob(
                segment,
                slot.blob_view(layout.num_ranks * segment.row_bytes),
                num_ranks=layout.num_ranks,
            )
            watchdog.check(layer, segment.segment, slot)
            if segment.blob_bytes:
                slot.blob_view(segment.blob_bytes)[segment.blob_bytes - 1] ^= 0x80
                watchdog.check(layer, segment.segment, slot)
    flipped = [key for key, segment in layout.segments.items() if segment.blob_bytes]
    findings = watchdog.poll()
    assert [(layer, kind) for layer, kind, _ in findings] == flipped
    assert all(count == 1 for _, _, count in findings)
    assert watchdog.checks == len(layout.segments) + len(flipped)
