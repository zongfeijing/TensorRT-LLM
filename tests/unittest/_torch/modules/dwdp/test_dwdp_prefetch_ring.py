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
"""CPU tests of the expert prefetch ring's memory precheck and of the page pool's failure path.

``WeightBuffer.create`` compares the page pool of the remote-expert double buffer (the prefetch
ring) with the free device memory before ``PagePool`` allocates a page, and raises
``PrefetchRingMemoryError`` when it does not fit.  The device is replaced: the VMM granularity,
``torch.cuda.empty_cache``, ``torch.cuda.mem_get_info``, ``PagePool.create`` and the per-layer VA
mappings are patched.  When a page allocation fails, ``PagePool`` releases every handle it created once.
"""

import gc
import pickle
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest
import torch

from tensorrt_llm._torch.modules.dwdp import page_pool, weight_buffer
from tensorrt_llm._torch.modules.dwdp.page_pool import PagePool, compute_slot_sizes
from tensorrt_llm._torch.modules.dwdp.specs import MnnvlHandleSet, PageAlignedLayout, WeightSpec
from tensorrt_llm._torch.modules.dwdp.weight_buffer import PrefetchRingMemoryError, WeightBuffer

pytestmark = pytest.mark.cpu_only

MIB = 1 << 20
GIB = 1 << 30
GRANULARITY = 2 * MIB
PAGE = 16 * MIB
TOTAL = 278 * GIB


def _specs(layers, weights, num_experts, local_start, local_end):
    """``LayerWeightSpecs`` of uint8 weights; ``weights`` maps a name to its per-expert shape."""
    return {
        layer: {
            name: WeightSpec(
                num_experts=num_experts,
                chunk_shape=(local_end - local_start, *shape),
                full_shape=(num_experts, *shape),
                dtype=torch.uint8,
            )
            for name, shape in weights.items()
        }
        for layer in layers
    }


def _handles(specs, local_start, local_end):
    """One handle per ``(layer, weight)``, sized as the transport sizes it: the local span in pages."""
    handles, sizes = {}, {}
    for layer, weights in specs.items():
        for name, spec in weights.items():
            start = local_start * spec.expert_bytes // GRANULARITY * GRANULARITY
            end = -(-local_end * spec.expert_bytes // GRANULARITY) * GRANULARITY
            handles[(layer, name)] = len(handles) + 1
            sizes[(layer, name)] = end - start
    return MnnvlHandleSet(handles=handles, sizes=sizes)


def _slot_sizes(specs, local_start, local_end):
    """The ring's two slot sizes, from the layouts as the buffer lays them out."""
    layouts = {
        layer: {
            name: PageAlignedLayout.compute(
                expert_bytes=spec.expert_bytes,
                num_experts=spec.num_experts,
                local_start=local_start,
                local_end=local_end,
                granularity=GRANULARITY,
                handle_phys_size=1 << 62,
                pool_granularity=PAGE,
            )
            for name, spec in weights.items()
        }
        for layer, weights in specs.items()
    }
    slots = {layer: position % 2 for position, layer in enumerate(sorted(specs))}
    return compute_slot_sizes(layouts, slots)


class _Device:
    """The device under ``WeightBuffer.create``: ``free`` of ``total`` bytes free; records calls."""

    def __init__(self, stack: ExitStack, free: int, total: int = TOTAL) -> None:
        self.free = free
        self.total = total
        self.calls: list = []
        stack.enter_context(
            patch.object(weight_buffer, "get_allocation_granularity", return_value=GRANULARITY)
        )
        stack.enter_context(
            patch.object(
                weight_buffer.torch.cuda,
                "empty_cache",
                side_effect=lambda: self.calls.append("empty_cache"),
            )
        )
        stack.enter_context(
            patch.object(weight_buffer.torch.cuda, "mem_get_info", side_effect=self.mem_get_info)
        )
        self.create = stack.enter_context(
            patch.object(weight_buffer.PagePool, "create", return_value=MagicMock(name="pool"))
        )
        stack.enter_context(patch.object(weight_buffer.WeightBuffer, "_setup_layer"))

    def mem_get_info(self, device_id: int) -> tuple[int, int]:
        self.calls.append(("mem_get_info", device_id))
        return self.free, self.total


def _create(specs, local_start, local_end, dwdp_size, device_id=0) -> WeightBuffer:
    return WeightBuffer.create(
        layer_weight_specs=specs,
        handles=_handles(specs, local_start, local_end),
        local_start=local_start,
        local_end=local_end,
        dwdp_size=dwdp_size,
        device_id=device_id,
    )


def _small_specs():
    """Two layers of two weights; 16 experts of 1 MiB, 4 of them local: 2 pages per slot."""
    return _specs((3, 4), {"w1": (1024, 1024), "w2": (512, 2048)}, 16, 0, 4)


def test_a_ring_larger_than_free_memory_is_refused_before_any_allocation():
    specs = _small_specs()
    slot_sizes = _slot_sizes(specs, 0, 4)
    assert slot_sizes == [2 * PAGE, 2 * PAGE]
    needed = sum(slot_sizes)
    with ExitStack() as stack:
        device = _Device(stack, free=needed - 1)

        with pytest.raises(PrefetchRingMemoryError) as raised:
            _create(specs, 0, 4, dwdp_size=4)

        device.create.assert_not_called()
    assert device.calls == ["empty_cache", ("mem_get_info", 0)]
    error = raised.value
    assert isinstance(error, RuntimeError)
    numbers = (error.needed_bytes, error.free_bytes, error.total_bytes, error.device_id)
    assert numbers == (needed, needed - 1, TOTAL, 0)
    assert (error.slot_pages, error.page_size) == ((2, 2), PAGE)
    message = str(error)
    assert message.startswith(
        "[DWDP Setup] The expert prefetch ring does not fit on GPU 0: its two slots need 0.06 GiB "
        "(2 + 2 pages of 16 MiB, the remote experts of two MoE layers), but 0.06 GiB of 278.00 "
        "GiB is free."
    )
    assert "use a larger dwdp_size" in message
    # Dense DWDP is a remedy only where the decoder layers have dense roots (expert-only models
    # refuse TRTLLM_DWDP_DENSE=1).
    assert (
        "On a model with dense DWDP roots (Kimi-K3), dense DWDP (TRTLLM_DWDP_DENSE=1) releases the "
        "replicated dense weights before the ring is allocated" in message
    )
    # The executor sends a setup error to the proxy pickled: the error survives it.
    copy = pickle.loads(pickle.dumps(error))
    assert type(copy) is PrefetchRingMemoryError and str(copy) == message
    assert (copy.needed_bytes, copy.slot_pages, copy.page_size) == (needed, (2, 2), PAGE)


def test_a_page_pool_allocation_failure_releases_every_handle_once():
    """Slot 1 page 2 fails: the 5 handles made so far are released once, ``__del__`` adds none."""
    created, released = [], []

    def create_local_handle(size: int, device_id: int) -> int:
        assert (size, device_id) == (PAGE, 0)
        if len(created) == 5:
            raise RuntimeError("CUDA_ERROR_OUT_OF_MEMORY")
        created.append(1000 + len(created))
        return created[-1]

    with (
        patch.object(page_pool, "create_local_handle", side_effect=create_local_handle),
        patch.object(page_pool, "release_handle", side_effect=released.append),
    ):
        error = None
        try:
            PagePool([3 * PAGE, 3 * PAGE], 0, granularity=GRANULARITY, page_size=PAGE)
        except RuntimeError as exc:
            error = exc
        assert str(error) == "Failed to allocate page 2 in slot 1: CUDA_ERROR_OUT_OF_MEMORY"
        assert str(error.__cause__) == "CUDA_ERROR_OUT_OF_MEMORY"
        # Drop the traceback, the last reference to the half-built pool: its __del__ runs now.
        del error
        gc.collect()
    assert created == [1000, 1001, 1002, 1003, 1004]
    assert sorted(released) == created
