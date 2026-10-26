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
"""MegaMoE's generation-phase kernel on 4 GPUs: a rank that no route reaches resets in order.

The main kernel's tail zeroes the workspace counters for the next launch. A rank that owns FC1 or
FC2 tiles reaches its tail only after its tiles, which wait for every peer's payload. A rank that
no route reaches owns no tile, so its tail used to follow only its own ``metadata_ready``. With
the main grid resident beside the communication kernel's refine CTAs (an SM reserve, as the dense
DWDP SM gather sets), nothing else ordered the reset: it could zero ``token_data_ready`` while a
peer's pushers had yet to publish to this rank. The late flags then left a residue, and the
Gather-B warps and refine CTAs waiting for the exact target never returned. That rank never
launched the next layer's communication kernel, and every rank hung.

The test forces that schedule, with dense DWDP off. Four processes, one per GPU, build an EP4 MoE
on ``MEGAMOE_CUTEDSL`` with the generation-phase kernel (32 tokens per rank, its 32-token bucket)
at SM reserve 46, which keeps the main grid resident beside the communication grid. A test-only
subclass of the communication component, swapped in by each process before any kernel exists
(``_megamoe_genphase_race_worker.py``), holds rank 3's ``metadata_ready``, and so its tail, for
300 us, and rank 0's payload flags for 1500 us. After a warm-up forward in which every rank
receives routes, four forwards route nothing to rank 3: its tail lands once ranks 1 to 3 have
published to it and before rank 0 does. With a tail that waits only for ``metadata_ready``, every
rank hangs in them; with the tail waiting for the publications, its own Gather-B waits and its
refine claims, the forwards finish, match the NVFP4 reference and agree bit for bit. A last
forward routes to every rank again, so a counter the zero-route tails left behind (a refine claim
that landed after the reset, say) shows up there, as a hang or a mismatch.

A hang is detected, not waited out: the processes run in their own sessions, and once all of
them are past the warm-up (compilation included), they get ``_RACE_TIMEOUT_S`` to finish before
they are killed and the test fails.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_NUM_RANKS = 4
_WORKER = Path(__file__).with_name("_megamoe_genphase_race_worker.py")
#: Compilation of the MegaMoE kernels, weight setup and the warm-up forward, per process.
_SETUP_TIMEOUT_S = 1800
#: The race forwards are short; anything near this is a hang.
_RACE_TIMEOUT_S = 120


def _skip_reason() -> str | None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < _NUM_RANKS:
        return f"needs {_NUM_RANKS} GPUs"
    from tensorrt_llm._torch.moe.fused_moe.mega_moe import mega_moe_cute_dsl
    from tensorrt_llm._utils import get_sm_version

    if get_sm_version() != 107:
        return "the generation-phase MegaMoE kernel is SM107-only"

    available, reason = mega_moe_cute_dsl.is_megamoe_cute_dsl_runtime_available()
    return None if available else reason


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _tail(path: Path, lines: int = 15) -> str:
    text = path.read_text(errors="replace") if path.exists() else ""
    return "\n".join(text.splitlines()[-lines:])


def _kill(procs: list[subprocess.Popen]) -> None:
    for proc in procs:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for proc in procs:
        proc.wait()


def _wait(procs, deadline_s, done) -> bool:
    """Poll until ``done()`` or a process fails; False when the deadline passes first."""
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        if done():
            return True
        if any(proc.poll() not in (None, 0) for proc in procs):
            return True
        time.sleep(1)
    return done()


@pytest.mark.gpu4
def test_zero_route_rank_tail_waits_for_late_publications(tmp_path: Path) -> None:
    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)

    port = _free_port()
    logs = [tmp_path / f"rank{rank}.log" for rank in range(_NUM_RANKS)]
    procs = []
    try:
        for rank in range(_NUM_RANKS):
            with logs[rank].open("w") as log:
                procs.append(
                    subprocess.Popen(
                        [sys.executable, str(_WORKER), str(rank), str(port), str(tmp_path)],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                )

        def marked(kind: str) -> bool:
            return all((tmp_path / f"rank{r}.{kind}.json").exists() for r in range(_NUM_RANKS))

        def report() -> str:
            return "\n".join(
                f"--- rank {r} (exit {procs[r].poll()}), {logs[r]}:\n{_tail(logs[r])}"
                for r in range(_NUM_RANKS)
            )

        if not _wait(procs, _SETUP_TIMEOUT_S, lambda: marked("warm")) or not marked("warm"):
            failed = [r for r in range(_NUM_RANKS) if procs[r].poll() not in (None, 0)]
            what = f"ranks {failed} failed" if failed else f"no warm-up within {_SETUP_TIMEOUT_S} s"
            pytest.fail(f"{what}\n{report()}")

        finished = _wait(
            procs, _RACE_TIMEOUT_S, lambda: all(proc.poll() is not None for proc in procs)
        )
        if not finished:
            stuck = [r for r in range(_NUM_RANKS) if procs[r].poll() is None]
            _kill(procs)
            pytest.fail(
                f"MegaMoE hang: ranks {stuck} did not finish the zero-route forwards within "
                f"{_RACE_TIMEOUT_S} s of the warm-up\n{report()}"
            )
        failed = [r for r in range(_NUM_RANKS) if procs[r].returncode != 0]
        assert not failed and marked("done"), f"ranks {failed} failed\n{report()}"
    finally:
        _kill(procs)
