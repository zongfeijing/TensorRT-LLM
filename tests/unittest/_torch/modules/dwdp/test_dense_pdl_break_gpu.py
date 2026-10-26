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
"""The PDL break of dense DWDP in a captured CUDA graph, on one GPU.

A closed-protocol ``DenseDWDP`` over four covered layers runs on ``TorchCudaOps`` with its slot
rings on the GPU; its launcher stands in for the SM gather with a spin kernel on the copy stream,
or copies with the copy engines (one ``cudaMemcpyBatchAsync`` per layer, issued as the batched
transfer issues it).  After H1 and after H2 of every layer the forward runs Kimi-K3's persistent
attention-residual kernel, which TensorRT-LLM launches with programmatic dependent launch (PDL),
as the decoder layer does above 32 tokens per rank.  The test captures a forward and reads the
nodes and edges of the CUDA graph:

* With the PDL break, the only dependency of every attention-residual kernel is one break, a
  kernel node of its own (the ``fill_`` of a one-element buffer, launched without PDL), and every
  dependency of a break is a full one: at H1 on the gather that the hook waited for, at H2 on the
  attention-residual kernel before it.  No gather depends on a break, so the gather that becomes
  eligible at a hook is launched together with the break, ahead of the next PDL kernel.  The
  compute nodes form one chain, break and kernel by turns, two of each per layer.
* Without it, the negative control: the attention-residual kernel after H1 depends on the gather
  it waited for through a programmatic edge, so it can launch once the gather's blocks have
  exited, ahead of the next gather, which waits for the gather's completion.
* With the copy engines the break is off by default (``DenseConfig.pdl_break_enabled``), and it
  is not needed: every edge from a copy node into an attention-residual kernel is a full one.
  Turned on (``TRTLLM_DWDP_DENSE_PDL_BREAK=1``), the break chain is the same as with the gather.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import NamedTuple

import pytest
import torch

from tensorrt_llm._torch.modules.dwdp.dense import plan as dense_plan
from tensorrt_llm._torch.modules.dwdp.dense.config import DenseConfig
from tensorrt_llm._torch.modules.dwdp.dense.runtime import DenseDWDP
from tensorrt_llm._torch.modules.dwdp.dense.types import (
    BATCH_SLICE_BYTES,
    RING_DEPTH,
    ROW_ALIGN,
    DenseProtocol,
    PrimePolicy,
    SegmentKind,
    SlotSpec,
    TorchCudaOps,
    allocate_slot_ring,
)

LAYERS = (1, 2, 3, 4)
SEGMENTS = (SegmentKind.ATTN, SegmentKind.MOE)
#: Tokens and hidden size of the attention-residual mix: the persistent kernel's shape.
TOKENS = 48
HIDDEN = 7168
#: GPU clock cycles of a gather's spin kernel.
SPIN_CYCLES = 10_000
#: Pieces of a copy-engine launch, each of the batched transfer's piece size.
COPY_PIECES = 4

#: ``cudaGraphDependencyType``: a full dependency, and a programmatic one (PDL).
FULL = 0
PROGRAMMATIC = 1

#: Node kinds: kernels from their names, and the copy nodes of the copy engines.
GATHER, BREAK, MIX, COPY = "gather", "break", "mix", "copy"
_KINDS = {"spin_kernel": GATHER, "FillFunctor": BREAK, "attn_res_fwd_persistent": MIX}

#: The launchers: the SM gather's stand-in, and the copy engines.
SPIN_GATHER, COPY_ENGINES = "spin_gather", "copy_engines"
#: The node kind of each launcher's copies.
_COPY_KIND = {SPIN_GATHER: GATHER, COPY_ENGINES: COPY}


class Graph(NamedTuple):
    """The nodes of a captured graph, by kind, and its edges ``(source, target) -> type``."""

    kinds: dict[int, str]
    edges: dict[tuple[int, int], int]

    def nodes(self, kind: str) -> list[int]:
        return [node for node, node_kind in self.kinds.items() if node_kind == kind]

    def into(self, node: int) -> dict[int, int]:
        """The sources of the edges into ``node``, with the edge types."""
        return {source: kind for (source, target), kind in self.edges.items() if target == node}


def _check(result: tuple) -> object:
    """The values of a ``cuda.bindings`` call after its error code, which must be success."""
    error, *values = result
    assert int(error) == 0, f"CUDA error {error}"
    return values[0] if len(values) == 1 else values


def _read_graph(captured: torch.cuda.CUDAGraph) -> Graph:
    """The kernel nodes of a graph captured with ``keep_graph=True``, and its typed edges."""
    from cuda.bindings import driver, runtime

    graph = runtime.cudaGraph_t(init_value=captured.raw_cuda_graph())
    _, count = _check(runtime.cudaGraphGetNodes(graph, numNodes=0))
    nodes, count = _check(runtime.cudaGraphGetNodes(graph, numNodes=count))
    kinds = {}
    for node in nodes[:count]:
        try:
            _, node_type = runtime.cudaGraphNodeGetType(node)
        except ValueError:  # a node type this cuda.bindings has no enum member for
            node_type = None
        if node_type is None:
            # A node type that this CUDA runtime or cuda.bindings does not name (the query fails):
            # on some drivers a captured cudaMemcpyBatchAsync is one node of such a type.
            kinds[int(node)] = COPY
            continue
        kind = f"other:{node_type}"
        if node_type == runtime.cudaGraphNodeType.cudaGraphNodeTypeMemcpy:
            kind = COPY
        if node_type == runtime.cudaGraphNodeType.cudaGraphNodeTypeKernel:
            params = _check(driver.cuGraphKernelNodeGetParams(driver.CUgraphNode(int(node))))
            name = _check(driver.cuFuncGetName(params.func))
            name = name.decode() if isinstance(name, bytes) else str(name)
            kind = next((kind for key, kind in _KINDS.items() if key in name), f"kernel:{name}")
        kinds[int(node)] = kind
    get_edges = getattr(runtime, "cudaGraphGetEdges_v2", runtime.cudaGraphGetEdges)
    values = _check(get_edges(graph, numEdges=0))
    if len(values) != 4:
        pytest.skip("this cuda.bindings returns no edge types")
    sources, targets, data, count = _check(get_edges(graph, numEdges=values[-1]))
    edges = {
        (int(sources[index]), int(targets[index])): int(data[index].type) for index in range(count)
    }
    return Graph(kinds, edges)


@pytest.fixture(scope="module")
def device() -> Iterator[torch.device]:
    """A GPU with the persistent attention-residual kernel and PDL on; skips otherwise."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if os.environ.get("TRTLLM_ENABLE_PDL") == "0":
        pytest.skip("TRTLLM_ENABLE_PDL=0: no kernel is launched with PDL")
    pytest.importorskip("cuda.bindings")
    try:
        torch.cuda.CUDAGraph(keep_graph=True)
    except TypeError:
        pytest.skip("this torch cannot keep a captured graph (CUDAGraph(keep_graph=True))")
    if not hasattr(torch.ops.trtllm, "attn_res_add_rmsnorm_persistent_fwd"):
        pytest.skip("this build has no persistent attention-residual kernel")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the persistent attention-residual kernel needs an sm_10x GPU")
    yield torch.device("cuda", torch.cuda.current_device())


class _CopyEngineLauncher:
    """Copies ``COPY_PIECES`` pieces with one ``cudaMemcpyBatchAsync`` per layer on the current
    stream, as the batched transfer (``DenseCopyLauncher`` with ``CopyMode.BATCHED``) does."""

    def __init__(self, device: torch.device) -> None:
        from cuda.bindings import runtime as cudart

        error = dense_plan.batched_copy_capture_error(device)
        if error is not None:
            pytest.skip(f"this driver cannot capture the batched copies: {error}")
        self._cudart = cudart
        self._attrs = [dense_plan._batched_copy_attributes(cudart)]
        self._device = device
        nbytes = COPY_PIECES * BATCH_SLICE_BYTES
        self._src = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device=device)
        self._dst = torch.empty_like(self._src)
        offsets = [piece * BATCH_SLICE_BYTES for piece in range(COPY_PIECES)]
        self._copies = (
            [self._dst.data_ptr() + offset for offset in offsets],
            [self._src.data_ptr() + offset for offset in offsets],
            [BATCH_SLICE_BYTES] * COPY_PIECES,
        )

    def __call__(self, layer: int) -> None:
        dense_plan._memcpy_batch(self._cudart, self._attrs, self._device, layer, self._copies)


def _capture(device: torch.device, pdl_break: bool, launcher: str = SPIN_GATHER) -> Graph:
    """An eager forward, then a forward captured into a CUDA graph and replayed; its graph."""
    # torch.cuda.Stream() hands out the streams of a pool of 32 by turns, so a new copy stream may
    # be the default capture stream of torch.cuda.graph; the capture runs on the stream after it.
    copy_stream, capture_stream = torch.cuda.Stream(device), torch.cuda.Stream(device)
    assert copy_stream.cuda_stream != capture_stream.cuda_stream
    slots = {
        kind: allocate_slot_ring(SlotSpec(kind, RING_DEPTH[kind], ROW_ALIGN, 2), device)
        for kind in SEGMENTS
    }
    runtime = DenseDWDP(
        LAYERS,
        SEGMENTS,
        slots,
        copy_stream,
        ops=TorchCudaOps(device),
        protocol=DenseProtocol.CLOSED,
        prime=PrimePolicy.WRAP,
        pdl_break=pdl_break,
    )
    if launcher == COPY_ENGINES:
        runtime.set_launcher(_CopyEngineLauncher(device))
    else:
        runtime.set_launcher(lambda layer: torch.cuda._sleep(SPIN_CYCLES))
    runtime.commit(None, fill_byte=0)
    prefix = torch.randn(TOKENS, 1, HIDDEN, dtype=torch.bfloat16, device=device)
    snapshots = torch.randn(1, TOKENS, 1, HIDDEN, dtype=torch.bfloat16, device=device)
    weight = torch.ones(HIDDEN, dtype=torch.bfloat16, device=device)

    def mix() -> None:
        torch.ops.trtllm.attn_res_add_rmsnorm_persistent_fwd(
            prefix, None, snapshots, weight, weight, weight, 1e-6, 1e-6
        )

    def forward() -> None:
        runtime.begin_forward()
        for layer in LAYERS:
            runtime.wait_layer(layer)  # H1
            mix()
            runtime.attention_done(layer)  # H2
            mix()
            runtime.layer_done(layer)  # H3
        runtime.end_forward(True)

    try:
        forward()  # the first forward after commit, which primes under wrap
        captured = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(captured, stream=capture_stream):
            forward()
        captured.replay()
        torch.cuda.synchronize(device)
        assert (runtime.captures, runtime.recoveries, runtime.demand_fetches) == (1, 0, 0)
        return _read_graph(captured)
    finally:
        runtime.release()


def _chain(graph: Graph, nodes: list[int]) -> list[int]:
    """``nodes`` in the order of the one path through all of them, which must exist."""
    inside = set(nodes)
    successors = {node: [] for node in nodes}
    for source, target in graph.edges:
        if source in inside and target in inside:
            successors[source].append(target)
    first = [node for node in nodes if not inside.intersection(graph.into(node))]
    assert len(first) == 1, f"the compute nodes start at {len(first)} nodes"
    chain = first
    while successors[chain[-1]]:
        (following,) = successors[chain[-1]]
        chain.append(following)
    assert sorted(chain) == sorted(nodes), "the compute nodes do not form one chain"
    return chain


@pytest.mark.parametrize("launcher", [SPIN_GATHER, COPY_ENGINES])
def test_the_pdl_break_sits_between_each_hook_and_its_pdl_kernel(device, launcher):
    """Each PDL kernel depends only on its hook's break; the breaks only on full dependencies.

    With the copy engines the break runs only when ``TRTLLM_DWDP_DENSE_PDL_BREAK=1`` turns it on;
    the chain is the same.
    """
    copy = _COPY_KIND[launcher]
    graph = _capture(device, pdl_break=True, launcher=launcher)
    mixes, breaks, gathers = graph.nodes(MIX), graph.nodes(BREAK), graph.nodes(copy)
    assert (len(mixes), len(breaks)) == (2 * len(LAYERS), 2 * len(LAYERS)), graph.kinds
    assert sorted(set(graph.kinds.values())) == sorted({copy, BREAK, MIX}), graph.kinds
    chain = _chain(graph, mixes + breaks)
    assert [graph.kinds[node] for node in chain] == [BREAK, MIX] * (2 * len(LAYERS))
    for mix in mixes:
        sources = sorted((graph.kinds[source], edge) for source, edge in graph.into(mix).items())
        assert sources == [(BREAK, PROGRAMMATIC)], f"a PDL kernel depends on {sources}"
    for number, brk in enumerate(chain[::2]):
        sources = graph.into(brk)
        assert set(sources.values()) <= {FULL}, f"break {number} has a programmatic dependency"
        waited = [source for source in sources if graph.kinds[source] == copy]
        if number % 2 == 0 and number >= 4:  # H1 of a layer copied in the capture: L3, L4
            assert waited, f"the break of H1(L{LAYERS[number // 2]}) does not follow its gather"
        if number % 2 == 1:  # H2: after the attention-residual kernel, before the gather
            assert waited == [] and chain[2 * number - 1] in sources
    for gather in gathers:
        waited = {graph.kinds[source] for source in graph.into(gather)}
        assert BREAK not in waited, f"a gather waits for a break: {waited}"


def test_without_the_pdl_break_the_kernel_after_h1_launches_with_the_gather_it_waits_for(device):
    """The negative control: the edge the PDL break removes."""
    graph = _capture(device, pdl_break=False)
    assert graph.nodes(BREAK) == []
    programmatic = [
        (source, mix)
        for mix in graph.nodes(MIX)
        for source, kind in graph.into(mix).items()
        if graph.kinds[source] == GATHER and kind == PROGRAMMATIC
    ]
    assert len(programmatic) >= 2, graph.edges  # H1 of L3 and L4 wait for gathers of the capture


def test_with_the_copy_engines_the_pdl_kernels_depend_on_the_copies_through_full_edges(device):
    """The copy engines' default: no break, and no programmatic edge from a copy to a PDL kernel.

    A copy node holds no SMs, and a programmatic dependency joins kernel nodes only, so the PDL
    kernel after H1 starts once the copy it waited for has completed.
    """
    pdl_break = DenseConfig().pdl_break_enabled(use_gather=False, closed=True)
    assert not pdl_break
    graph = _capture(device, pdl_break=pdl_break, launcher=COPY_ENGINES)
    assert graph.nodes(BREAK) == []
    assert sorted(set(graph.kinds.values())) == sorted({COPY, MIX}), graph.kinds
    # The capture copies L3 and L4 at H2 and, wrapping, L1 and L2 at H3: one batched copy each.
    assert len(graph.nodes(COPY)) >= 4, graph.kinds
    copy_edges = [
        kind
        for mix in graph.nodes(MIX)
        for source, kind in graph.into(mix).items()
        if graph.kinds[source] == COPY
    ]
    assert len(copy_edges) >= 2, graph.edges  # H1 of L3 and L4 wait for copies of the capture
    assert set(copy_edges) == {FULL}, graph.edges
