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
"""Tests for the dense DWDP row store and its release.

The layouts come from toy decoder layers through ``layout.collect_dense_layout``: each layer's
attention and MoE-dense roots hold parameters of the given dtypes and shapes, filled with seeded
random bytes.  The big layers have rows of about 1.25 MiB (``N = 4``), so
``STORE_CHUNK_MB`` splits them.

The CPU tests (``cpu_only``) check that:

* the FABRIC store, over a recording fake of the CUDA VMM calls, maps its chunks read-write and
  zero-filled, writes this rank's rows, exports one handle per chunk, imports and maps every
  peer's handles read-only in rank order, and releases peer mappings, imported handles, local
  mappings and local handles in that order -- also after a failure at any step of
  ``allocate`` or ``import_peers``, which then does not run again.
"""

import ctypes
import math
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.modules.dwdp import transport, vmm
from tensorrt_llm._torch.modules.dwdp.dense import store as dense_store
from tensorrt_llm._torch.modules.dwdp.dense.config import DenseConfig
from tensorrt_llm._torch.modules.dwdp.dense.layout import collect_dense_layout, write_blob
from tensorrt_llm._torch.modules.dwdp.dense.store import FabricShardStore
from tensorrt_llm._torch.modules.dwdp.dense.types import ROW_ALIGN, DenseLayout

CPU = torch.device("cpu")
BF16 = torch.bfloat16
U8 = torch.uint8

FABRIC = vmm.cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
PROT_READ = vmm.cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READ
PROT_READWRITE = vmm.cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

#: Storages below this stay replicated in the toy layouts.
TOY_MIN_BYTES = 64

#: Four layers whose rows take about 1.25 MiB each with N = 4 (A 3 MiB + 5 * l bytes, D 2 MiB).
BIG_LAYERS = {
    layer: ([(U8, ((3 << 20) + 5 * layer,))], [(BF16, (1 << 20,))]) for layer in range(1, 5)
}
GARBAGE = 0xEE  # Fake device memory before the store zero-fills it.


# ---------------------------------------------------------------------------------------------
# Toy layers and helpers
# ---------------------------------------------------------------------------------------------


class ToyRoot(nn.Module):
    """A dense root holding one Parameter per ``(dtype, shape)``, filled with random bytes."""

    def __init__(self, tensors, generator: torch.Generator, device: torch.device) -> None:
        super().__init__()
        for index, (dtype, shape) in enumerate(tensors):
            nbytes = math.prod(shape) * dtype.itemsize
            data = torch.randint(0, 256, (nbytes,), generator=generator, dtype=torch.uint8)
            weight = data.view(dtype).reshape(shape).to(device)
            self.register_parameter(f"w{index}", nn.Parameter(weight, requires_grad=False))


class ToyLayer(nn.Module):
    """A decoder layer with the ``dense_dwdp_roots`` seam; ``spec`` None keeps it replicated."""

    def __init__(self, spec, generator: torch.Generator, device: torch.device) -> None:
        super().__init__()
        self.covered = spec is not None
        attn, moe = spec if spec is not None else ([], [])
        self.attn = ToyRoot(attn, generator, device)
        self.moe = ToyRoot(moe, generator, device)

    def dense_dwdp_roots(self, segment: str) -> list[tuple[str, nn.Module]]:
        return [(segment, getattr(self, segment))] if self.covered else []


def collect(
    spec, num_ranks: int, device: torch.device = CPU, *, segments=("attn", "moe"), chunk_mb=0
) -> tuple[DenseLayout, list[ToyLayer]]:
    """The layout of toy layers built from ``spec`` (seeded, so every call gives equal bytes)."""
    generator = torch.Generator().manual_seed(1234)
    layers = [(index, ToyLayer(spec[index], generator, device)) for index in sorted(spec)]
    config = DenseConfig(
        enabled=True, segments=segments, min_bytes=TOY_MIN_BYTES, store_chunk_mb=chunk_mb
    )
    layout = collect_dense_layout(
        layers,
        config,
        num_ranks=num_ranks,
        device=device,
        attr_allow=frozenset(),
        exclude=frozenset(),
    )
    return layout, [layer for _, layer in layers]


def blob(segment, num_ranks: int, device: torch.device = CPU) -> torch.Tensor:
    """The full ``N * S`` blob of a segment, as a gather leaves it in a slot."""
    out = torch.empty(num_ranks * segment.row_bytes, dtype=torch.uint8, device=device)
    write_blob(segment, out, num_ranks=num_ranks)
    return out


def host_bytes(address: int, nbytes: int) -> torch.Tensor:
    """A copy of ``nbytes`` of host memory at ``address``."""
    if nbytes == 0:
        return torch.empty(0, dtype=torch.uint8)
    view = torch.frombuffer((ctypes.c_ubyte * nbytes).from_address(address), dtype=torch.uint8)
    return view.clone()


class RecordingLogger:
    """Stands in for the store module's logger and keeps every ``(level, message)``."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def debug(self, message: str) -> None:
        self.records.append(("debug", message))

    def info(self, message: str) -> None:
        self.records.append(("info", message))

    def warning(self, message: str) -> None:
        self.records.append(("warning", message))

    def error(self, message: str) -> None:
        self.records.append(("error", message))

    def messages(self, level: str) -> list[str]:
        return [message for name, message in self.records if name == level]


@pytest.fixture
def store_log(monkeypatch) -> RecordingLogger:
    """What the store logs during the test."""
    log = RecordingLogger()
    monkeypatch.setattr(dense_store, "logger", log)
    return log


# ---------------------------------------------------------------------------------------------
# CPU: the FABRIC store over a recording fake of the CUDA VMM calls
# ---------------------------------------------------------------------------------------------


class FakeVmm:
    """The VMM, export/import and synchronize calls of the FABRIC store, over host memory.

    Every call is recorded as ``(op, value)``; ``cuMemSetAccess`` is faked at the driver binding,
    so the access flags the store asks for are what the ``access_rw`` / ``access_ro`` records
    say.  Reserved address ranges are real host buffers filled with garbage (as fresh device
    memory may be), so the store's views alias them and its writes land there; with
    ``aliasing_views=False`` a view is a copy instead.

    Like the driver, the fake knows the size of every handle: created, or imported from a
    payload whose size ``export`` (or ``peer_payloads``, for a peer's) registered.  Handles and
    reservations are whole granules of ``granularity``, at least one, and a reservation must be
    aligned to it; a mapping covers one whole handle (a FABRIC handle cannot be mapped in part)
    and exactly its reservation; access, unmap and free name a whole reservation.  Reservations
    start ``reserve_skew`` bytes past a ``ROW_ALIGN`` boundary.  ``fail=(op, n)`` makes the
    ``n``-th call of ``op`` raise before it takes effect, ``fail_every`` every call of the ops it
    names.
    """

    GRANULARITY = 2 << 20

    def __init__(
        self,
        monkeypatch,
        *,
        fail=None,
        fail_every=frozenset(),
        handle_type=FABRIC,
        aliasing_views=True,
        granularity=GRANULARITY,
        reserve_skew=0,
    ) -> None:
        self.calls: list[tuple[str, object]] = []
        self.fail = fail
        self.fail_every = fail_every
        self.aliasing_views = aliasing_views
        self.granularity = granularity
        self.reserve_skew = reserve_skew
        self.counts: Counter = Counter()
        self.memory: dict[int, torch.Tensor] = {}
        self.owner: dict[int, int] = {}  # va -> the handle mapped (or about to be) there
        self.imported: set[int] = set()
        self.handle_sizes: dict[int, int] = {}
        self.payload_sizes: dict[bytes, int] = {}
        self.va_sizes: dict[int, int] = {}
        self._next_handle = 1000
        self._last_handle = None
        patches = {
            (vmm, "peer_handle_type"): lambda: handle_type,
            (vmm, "get_allocation_granularity"): lambda device_id: self.granularity,
            (vmm, "create_fabric_handle"): self.create,
            (vmm, "reserve_va"): self.reserve,
            (vmm, "map_handle"): self.map,
            (vmm.cuda, "cuMemSetAccess"): self.set_access,
            (vmm, "tensor_from_ptr"): self.view,
            (vmm, "unmap_va"): self.unmap,
            (vmm, "free_va"): self.free,
            (vmm, "release_handle"): self.release,
            (transport, "_export_handle"): self.export,
            (transport, "_import_handle"): self.import_handle,
            (torch.cuda, "synchronize"): self.synchronize,
            (torch.cuda, "current_stream"): self.current_stream,
        }
        for (owner, name), value in patches.items():
            monkeypatch.setattr(owner, name, value)

    def _record(self, op: str, value: object) -> None:
        self.counts[op] += 1
        if self.fail == (op, self.counts[op]) or op in self.fail_every:
            raise RuntimeError(f"injected failure of {op} #{self.counts[op]}")
        self.calls.append((op, value))

    def _new_handle(self) -> int:
        self._next_handle += 1
        return self._next_handle

    def _check_whole_granules(self, what: str, size: int) -> None:
        assert size > 0 and size % self.granularity == 0, f"{what} of {size} bytes"

    def _check_reservation(self, what: str, va: int, size: int) -> None:
        assert va in self.va_sizes, f"{what} at {va:#x}, which is not reserved"
        assert size == self.va_sizes[va], (
            f"{what} of {size} bytes at {va:#x}, whose reservation has {self.va_sizes[va]}"
        )

    def create(self, size, device_id):
        self._check_whole_granules("cuMemCreate", size)
        handle = self._new_handle()
        self._record("create", (handle, size))
        self.handle_sizes[handle] = size
        self._last_handle = handle
        return handle

    def import_handle(self, payload, handle_type):
        assert handle_type == FABRIC and isinstance(payload, bytes)
        assert payload in self.payload_sizes, f"import of an unknown payload {payload!r}"
        handle = self._new_handle()
        self._record("import", (handle, payload))
        self.handle_sizes[handle] = self.payload_sizes[payload]
        self.imported.add(handle)
        self._last_handle = handle
        return handle

    def export(self, handle, handle_type):
        assert handle_type == FABRIC
        self._record("export", handle)
        payload = f"fabric-handle-{handle}".encode()
        self.payload_sizes[payload] = self.handle_sizes[handle]
        return payload

    def reserve(self, size, alignment):
        assert alignment == self.granularity
        self._check_whole_granules("cuMemAddressReserve", size)
        self._record("reserve", size)
        buffer = torch.full((size + 2 * ROW_ALIGN,), GARBAGE, dtype=torch.uint8)
        va = buffer.data_ptr() + (-buffer.data_ptr()) % ROW_ALIGN + self.reserve_skew
        self.memory[va] = buffer
        self.va_sizes[va] = size
        self.owner[va] = self._last_handle
        self.calls[-1] = ("reserve", (va, size))
        return va

    def map(self, va, size, handle, offset=0):
        assert self.owner[va] == handle and offset == 0
        self._check_reservation("cuMemMap", va, size)
        assert size == self.handle_sizes[handle], (
            f"cuMemMap of {size} bytes of handle {handle}, which has {self.handle_sizes[handle]}"
        )
        self._record("map", (va, handle))

    def set_access(self, va, size, descriptors, count):
        """``cuMemSetAccess``: records ``access_rw`` or ``access_ro`` by the requested flags."""
        assert count == len(descriptors) == 1 and descriptors[0].location.id == 0
        self._check_reservation("cuMemSetAccess", va, size)
        flags = int(descriptors[0].flags)
        op = {int(PROT_READWRITE): "access_rw", int(PROT_READ): "access_ro"}.get(flags)
        assert op is not None, f"unexpected access flags {flags}"
        self._record(op, va)
        return (vmm.cuda.CUresult.CUDA_SUCCESS,)

    def view(self, ptr, shape, dtype, device_id):
        self._record("view", ptr)
        if not self.aliasing_views:
            return torch.zeros(shape, dtype=dtype)
        nbytes = math.prod(shape) * dtype.itemsize
        return host_bytes_view(ptr, nbytes).view(dtype).reshape(shape)

    def unmap(self, va, size):
        self._check_reservation("cuMemUnmap", va, size)
        self._record("unmap", va)

    def free(self, va, size):
        self._check_reservation("cuMemAddressFree", va, size)
        self._record("free", va)
        del self.memory[va]
        del self.va_sizes[va]

    def release(self, handle):
        self._record("release", handle)

    def synchronize(self, device=None):
        self._record("sync", device)

    def current_stream(self, device=None):
        return SimpleNamespace(synchronize=lambda: self._record("stream_sync", device))

    # Views of what happened ------------------------------------------------------------------

    def ops(self, start: int = 0) -> list[str]:
        return [op for op, _ in self.calls[start:]]

    def values(self, op: str, start: int = 0, stop: int | None = None) -> list:
        return [value for name, value in self.calls[start:stop] if name == op]

    def is_peer(self, value) -> bool:
        """Whether a recorded handle or address belongs to an imported (peer) handle."""
        return self.owner.get(value, value) in self.imported


def host_bytes_view(address: int, nbytes: int) -> torch.Tensor:
    """A ``uint8`` tensor that aliases ``nbytes`` of host memory at ``address``."""
    return torch.frombuffer((ctypes.c_ubyte * nbytes).from_address(address), dtype=torch.uint8)


def peer_payloads(fake: FakeVmm, store: FabricShardStore, layout: DenseLayout, rank: int):
    """Collective #2 as this rank sees it: its own export plus a distinct export per peer.

    Every rank allocates the same chunks, so a peer's handle of a chunk has the size of this
    rank's handle of it; ``fake`` learns that size for each fabricated payload.
    """
    own = store.export()
    payloads = []
    for peer in range(layout.num_ranks):
        if peer == rank:
            payloads.append(own)
            continue
        blobs = [f"peer-{peer}-chunk-{chunk.index}".encode() for chunk in layout.chunks]
        for blob, own_blob in zip(blobs, own):
            fake.payload_sizes[blob] = fake.payload_sizes[own_blob]
        payloads.append(blobs)
    return payloads


def check_release(fake: FakeVmm, start: int) -> None:
    """The calls since ``start`` undo every handle and mapping created before it, once each.

    Order: a device synchronize first; then the peer mappings (unmap, free) and imported
    handles, then the local mappings and handles; a mapping is unmapped before its VA is freed
    and both happen before its handle is released.
    """
    created = [handle for handle, _ in fake.values("create", 0, start)]
    imported = [handle for handle, _ in fake.values("import", 0, start)]
    reserved = [va for va, _ in fake.values("reserve", 0, start)]
    mapped = [va for va, _ in fake.values("map", 0, start)]
    tail = fake.calls[start:]
    if not (created or imported):
        assert tail == []
        return
    assert tail[0][0] == "sync"
    released = fake.values("release", start)
    assert sorted(released) == sorted(created + imported) and len(set(released)) == len(released)
    assert sorted(fake.values("free", start)) == sorted(reserved)
    assert sorted(fake.values("unmap", start)) == sorted(mapped)
    position = {(op, value): index for index, (op, value) in enumerate(tail)}
    for va in reserved:
        handle_released = position[("release", fake.owner[va])]
        assert position[("free", va)] < handle_released
        if va in mapped:
            assert position[("unmap", va)] < position[("free", va)]
    peer_steps = [i for i, (op, value) in enumerate(tail[1:]) if fake.is_peer(value)]
    local_steps = [i for i, (op, value) in enumerate(tail[1:]) if not fake.is_peer(value)]
    if peer_steps and local_steps:
        assert max(peer_steps) < min(local_steps), "peer resources must go before local ones"
    released_peers = [handle for handle in released if handle in fake.imported]
    peer_frees = [position[("free", va)] for va in reserved if fake.is_peer(va)]
    if released_peers and peer_frees:
        first_release = min(position[("release", handle)] for handle in released_peers)
        assert max(peer_frees) < first_release, (
            "peer VA ranges are freed before the imports are released"
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize("chunk_mb", [0, 1], ids=["one-chunk", "chunked"])
def test_fabric_store_maps_writes_exports_and_imports(monkeypatch, store_log, chunk_mb):
    """One handle per chunk mapped read-write and zeroed, this rank's rows, peers read-only."""
    num_ranks, rank = 4, 2
    layout, _ = collect(BIG_LAYERS, num_ranks, chunk_mb=chunk_mb)
    num_chunks = len(layout.chunks)
    fake = FakeVmm(monkeypatch)
    store = FabricShardStore(layout, rank, torch.device("cuda", 0))

    store.allocate()
    per_chunk = ["create", "reserve", "map", "access_rw", "view"]
    assert fake.ops() == per_chunk * num_chunks + ["stream_sync"]
    assert [size for _, size in fake.values("create")] == [c.handle_bytes for c in layout.chunks]
    assert store_log.messages("info") == []  # It allocated exactly ``layout.store_bytes``.
    local_vas = [va for va, _ in fake.values("reserve")]
    for chunk, va in zip(layout.chunks, local_vas):
        assert not host_bytes(va, chunk.handle_bytes).any(), f"chunk {chunk.index} not zeroed"

    store.write_local_rows()
    for chunk, va in zip(layout.chunks, local_vas):
        local = host_bytes(va, chunk.handle_bytes)
        written = torch.zeros_like(local)
        for layer in chunk.layers:
            for segment in layout.layer_segments(layer):
                width = segment.row_bytes
                row = blob(segment, num_ranks)[rank * width : (rank + 1) * width]
                written[segment.chunk_offset : segment.chunk_offset + width] = row
        assert torch.equal(local, written), f"chunk {chunk.index}"

    mark = len(fake.calls)
    payloads = peer_payloads(fake, store, layout, rank)
    assert payloads[rank] == [f"fabric-handle-{h}".encode() for h, _ in fake.values("create")]
    assert store.export() == payloads[rank]  # Exported once, returned again.
    assert fake.ops(mark) == ["export"] * num_chunks

    mark = len(fake.calls)
    store.import_peers(payloads)
    imports = [payload for _, payload in fake.values("import", mark)]
    assert imports == [blob for peer in (0, 1, 3) for blob in payloads[peer]]
    assert fake.ops(mark) == ["import", "reserve", "map", "access_ro"] * (3 * num_chunks)
    peer_sizes = [size for _, size in fake.values("reserve", mark)]
    assert peer_sizes == [chunk.handle_bytes for chunk in layout.chunks] * 3
    peer_vas = iter(va for va, _ in fake.values("reserve", mark))
    expected_bases = tuple(
        tuple(local_vas) if peer == rank else tuple(next(peer_vas) for _ in layout.chunks)
        for peer in range(num_ranks)
    )
    assert store.chunk_bases() == expected_bases

    mark = len(fake.calls)
    store.release()
    check_release(fake, mark)
    mark = len(fake.calls)
    store.release()
    assert fake.calls[mark:] == []
    assert fake.memory == {}


#: ``(op, n)`` failure points of ``allocate`` and ``import_peers`` on rank 1 of 3 with four chunks:
#: allocate makes four creates, reserves, maps, read-write accesses and views; import_peers makes
#: eight imports, reserves (calls 5-12 overall), maps and read-only accesses.
FAILURE_POINTS = [
    ("create", 1),
    ("create", 3),
    ("reserve", 2),
    ("map", 3),
    ("access_rw", 2),
    ("view", 4),
    ("stream_sync", 1),
    ("export", 2),
    ("import", 1),
    ("import", 6),
    ("reserve", 5),
    ("reserve", 11),
    ("map", 8),
    ("access_ro", 1),
    ("access_ro", 7),
]


@pytest.mark.cpu_only
@pytest.mark.parametrize("fail", FAILURE_POINTS, ids=[f"{op}-{n}" for op, n in FAILURE_POINTS])
def test_fabric_release_after_a_partial_allocate_or_import(monkeypatch, fail):
    """A step that fails part way does not run again and leaves what it created to ``release``."""
    num_ranks, rank = 3, 1
    layout, _ = collect(BIG_LAYERS, num_ranks, chunk_mb=1)
    assert len(layout.chunks) == 4
    fake = FakeVmm(monkeypatch, fail=fail)
    store = FabricShardStore(layout, rank, torch.device("cuda", 0))
    failed = "allocate"
    with pytest.raises(RuntimeError, match="injected failure"):
        store.allocate()
        failed = "write_local_rows"
        store.write_local_rows()
        failed = "export"
        payloads = peer_payloads(fake, store, layout, rank)
        failed = "import_peers"
        store.import_peers(payloads)
    with pytest.raises(RuntimeError):
        store.chunk_bases()
    if failed == "allocate":
        with pytest.raises(RuntimeError, match="allocate already ran"):
            store.allocate()
        for step in (store.write_local_rows, store.export):
            with pytest.raises(RuntimeError, match="before a successful allocate"):
                step()
    elif failed == "import_peers":
        with pytest.raises(RuntimeError, match="import_peers already ran"):
            store.import_peers(payloads)
    mark = len(fake.calls)
    store.release()
    check_release(fake, mark)
    assert fake.memory == {}
    mark = len(fake.calls)
    store.release()
    assert fake.calls[mark:] == []
