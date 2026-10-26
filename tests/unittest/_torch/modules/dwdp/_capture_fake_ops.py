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
"""Capture-aware recording fake of the dense DWDP CUDA seam (``types.DeviceOps``).

The recording model of ``test_dense_protocol`` (a wait binds to the latest record of its event, as
``cudaStreamWaitEvent`` does), plus the CUDA graph semantics the closed protocol relies on:

* :meth:`CaptureFakeOps.capture` captures on a capture stream of its own, as ``torch.cuda.graph``
  does.  A stream that waits for an event recorded in the capture joins it; operations on joined
  streams go into the graph instead of running.
* A wait inside the capture for an event whose latest record is outside it raises
  :class:`CaptureError` (``cudaErrorStreamCaptureIsolation``), and so does the end of a capture
  that a joined stream has not joined back (``cudaErrorStreamCaptureUnjoined``).
* An event whose latest record was made in a capture that has ended is invalid: an eager wait for
  it raises :class:`CaptureError` (``cudaErrorInvalidValue``) until it is recorded again.
* :meth:`CaptureFakeOps.replay` runs a graph again on the current stream: the capture stream's
  operations on the current stream, each joined stream's on a replay stream of its own, and every
  wait bound to the replay of the record it was bound to in the capture.

``log`` holds the operations that ran, in host order, as ``test_dense_protocol.Entry`` items, so
``test_dense_protocol.assert_race_free`` can check them.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from test_dense_protocol import Entry, happens_before, vector_clocks


class CaptureError(RuntimeError):
    """A CUDA error of graph capture: isolation, an unjoined capture or an invalid event."""


class Stream:
    """A named stream with the ``torch.cuda.Stream`` calls the runtime makes."""

    def __init__(self, ops: CaptureFakeOps, name: str) -> None:
        self.ops = ops
        self.name = name

    def wait_event(self, event: Event) -> None:
        self.ops.wait(self, event)

    def wait_stream(self, stream: Stream) -> None:
        event = Event(self.ops)
        event.record(stream)
        self.wait_event(event)

    def __repr__(self) -> str:
        return f"<stream {self.name}>"


class Event:
    """An event; its latest record is ``log[index]``, or ``graph.ops[index]`` in a capture."""

    def __init__(self, ops: CaptureFakeOps) -> None:
        self.ops = ops
        self.graph: Graph | None = None
        self.index: int | None = None

    def record(self, stream: Stream | None = None) -> None:
        self.ops.record(self, self.ops.current if stream is None else stream)


class Graph:
    """A captured graph: its operations in capture order, waits bound by index into them."""

    def __init__(self, origin: Stream) -> None:
        self.origin = origin
        self.streams = {origin}
        self.ops: list[Entry] = []


class CaptureFakeOps:
    """``DeviceOps`` over the streams ``exec``, ``copy``, ``aux`` and the capture stream."""

    def __init__(self) -> None:
        self.log: list[Entry] = []
        self.exec = Stream(self, "exec")
        self.copy = Stream(self, "copy")
        self.aux = Stream(self, "aux")
        self.side = Stream(self, "capture")
        self.current = self.exec
        self.multi_stream = False
        self.graph: Graph | None = None
        self.replay_streams: dict[Stream, Stream] = {}

    # -- DeviceOps -------------------------------------------------------------

    def new_event(self) -> Event:
        return Event(self)

    def current_stream(self) -> Stream:
        return self.current

    @contextlib.contextmanager
    def stream(self, stream: Stream) -> Iterator[Stream]:
        previous, self.current = self.current, stream
        try:
            yield stream
        finally:
            self.current = previous

    def synchronize(self) -> None:
        if self.graph is not None:
            raise CaptureError("cudaErrorStreamCaptureUnsupported: synchronize in a capture")
        self.log.append(Entry("sync"))

    def is_current_stream_capturing(self) -> bool:
        return self.capturing(self.current)

    def do_multi_stream(self) -> bool:
        return self.multi_stream

    @contextlib.contextmanager
    def nvtx_range(self, message: str) -> Iterator[None]:
        yield

    # -- stream operations -----------------------------------------------------

    def capturing(self, stream: Stream) -> bool:
        return self.graph is not None and stream in self.graph.streams

    def run(self, kind: str, stream: Stream, **fields: object) -> int:
        """Enqueue an operation on ``stream``: into the graph if it is capturing, else the log."""
        entry = Entry(kind, stream=stream, **fields)
        ops = self.graph.ops if self.capturing(stream) else self.log
        ops.append(entry)
        return len(ops) - 1

    def record(self, event: Event, stream: Stream) -> None:
        event.index = self.run("record", stream, event=event)
        event.graph = self.graph if self.capturing(stream) else None

    def wait(self, stream: Stream, event: Event) -> None:
        graph = self.graph
        if graph is not None and event.graph is graph:
            graph.streams.add(stream)  # waiting for a record of the capture joins it
        elif self.capturing(stream) and event.index is not None:
            raise CaptureError(
                f"cudaErrorStreamCaptureIsolation: {stream.name} waits in a capture for a "
                f"record made outside it"
            )
        elif event.graph is not None:
            raise CaptureError(
                f"cudaErrorInvalidValue: {stream.name} waits for an event last recorded in a "
                f"capture that has ended"
            )
        self.run("wait", stream, event=event, bound=event.index)

    # -- graphs ----------------------------------------------------------------

    @contextlib.contextmanager
    def capture(self) -> Iterator[Graph]:
        """``torch.cuda.graph``: capture until the block ends, also when it raises."""
        graph = self.graph = Graph(self.side)
        previous, self.current = self.current, self.side
        try:
            yield graph
        finally:
            self.current = previous
            self.graph = None
            clocks = vector_clocks(graph.ops)
            last = {entry.stream: index for index, entry in enumerate(graph.ops)}
            end = last.get(graph.origin)
            for stream, index in last.items():
                if stream is not graph.origin and (
                    end is None or not happens_before(graph.ops, clocks, index, end)
                ):
                    raise CaptureError(
                        f"cudaErrorStreamCaptureUnjoined: {stream.name} did not join the "
                        f"capture stream"
                    )

    def replay(self, graph: Graph) -> None:
        """``CUDAGraph.replay`` on the current stream."""
        assert self.graph is None, "replay inside a capture"
        placed: list[int] = []
        for entry in graph.ops:
            stream = self.current
            if entry.stream is not graph.origin:
                stream = self.replay_streams.get(entry.stream)
                if stream is None:
                    stream = Stream(self, f"replay:{entry.stream.name}")
                    self.replay_streams[entry.stream] = stream
            placed.append(len(self.log))
            self.log.append(
                Entry(
                    entry.kind,
                    stream=stream,
                    event=entry.event,
                    bound=None if entry.bound is None else placed[entry.bound],
                    what=entry.what,
                    layer=entry.layer,
                )
            )
