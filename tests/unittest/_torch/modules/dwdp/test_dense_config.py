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
"""CPU tests of the dense DWDP switches and how their values are parsed.

* Booleans: ``1/true/yes/on`` and ``0/false/no/off`` in any case and with blanks around them;
  unset or blank keeps the default; any other value logs a warning and keeps the default, so the
  default-off flags stay off and ``TRTLLM_DWDP_DENSE_REQUIRED`` fails safe (stays on).
* ``TRTLLM_DWDP_DENSE_SEGMENTS`` is a comma-separated, case-insensitive list that must include
  ``attn``; an invalid list raises.

The switch tests pass an ``environ`` dict and do not touch
the process environment.
"""

import pytest

from tensorrt_llm._torch.modules.dwdp.dense import config as dense_config
from tensorrt_llm._torch.modules.dwdp.dense.config import (
    DENSE_REQUIRED_ENV,
    DENSE_SEGMENTS_ENV,
    DenseConfig,
    parse_segments,
)

pytestmark = pytest.mark.cpu_only
JUNK_VALUES = ["2", "enable", "y", "onn", "-1"]


class _Warnings:
    """Stands in for the config module's ``logger``; keeps the warnings."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def warning(self, message: str) -> None:
        self.messages.append(message)


@pytest.fixture
def warnings(monkeypatch) -> list[str]:
    capture = _Warnings()
    monkeypatch.setattr(dense_config, "logger", capture)
    return capture.messages


def _env(name: str, value: str | None) -> dict[str, str]:
    """An environment with ``name`` set to ``value``, or empty when ``value`` is None."""
    return {} if value is None else {name: value}


@pytest.mark.parametrize(
    "value, required",
    [(None, True), ("", True), ("1", True), ("on", True), ("0", False), (" False ", False)],
)
def test_required_defaults_on(warnings, value, required):
    assert DenseConfig.from_env(_env(DENSE_REQUIRED_ENV, value)).required is required
    assert warnings == []


@pytest.mark.parametrize("value", JUNK_VALUES)
def test_an_unreadable_required_fails_safe(warnings, value):
    assert DenseConfig.from_env(_env(DENSE_REQUIRED_ENV, value)).required is True
    assert len(warnings) == 1 and warnings[0].endswith("using 1")


@pytest.mark.parametrize("value", ["moe", "atn", "attention", "attn,mlp", "attn;moe", ","])
def test_an_invalid_segment_list_raises(value):
    with pytest.raises(ValueError, match=f"{DENSE_SEGMENTS_ENV}={value!r} must list attn"):
        DenseConfig.from_env(_env(DENSE_SEGMENTS_ENV, value))
    with pytest.raises(ValueError, match="must list attn"):
        parse_segments(value)
