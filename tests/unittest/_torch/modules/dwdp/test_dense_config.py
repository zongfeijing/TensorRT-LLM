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
* ``TRTLLM_DWDP_DENSE_PRIME`` is read as set; ``DenseConfig.prime_policy`` parses it in any case,
  defaults to ``wrap`` and raises on any other value, while reading the switches never does.
* Each ``TRTLLM_DWDP_DENSE_DEBUG`` level adds to the one below: NaN slots, the slot watchdog, and
  at 3 the watchdog's device assert.

The switch tests pass an ``environ`` dict and do not touch
the process environment.
"""

import pytest

from tensorrt_llm._torch.modules.dwdp.dense import config as dense_config
from tensorrt_llm._torch.modules.dwdp.dense.config import (
    DENSE_DEBUG_ENV,
    DENSE_PRIME_ENV,
    DENSE_REQUIRED_ENV,
    DENSE_SEGMENTS_ENV,
    DenseConfig,
    parse_segments,
)
from tensorrt_llm._torch.modules.dwdp.dense.types import DEBUG_FILL_BYTE, PrimePolicy

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


@pytest.mark.parametrize(
    "value, policy",
    [
        (None, PrimePolicy.WRAP),
        (" ", PrimePolicy.WRAP),
        ("wrap", PrimePolicy.WRAP),
        (" Entry ", PrimePolicy.ENTRY),
        ("bogus", None),
        ("wrap,entry", None),
    ],
)
def test_prime_defaults_to_wrap_and_only_its_parse_rejects_junk(warnings, value, policy):
    config = DenseConfig.from_env(_env(DENSE_PRIME_ENV, value))
    assert warnings == []
    if policy is None:
        with pytest.raises(ValueError, match=f"{DENSE_PRIME_ENV}={value!r}"):
            config.prime_policy()
    else:
        assert config.prime_policy() is policy


@pytest.mark.parametrize(
    "level, fill_byte, watchdog, device_assert",
    [
        ("0", 0, False, False),
        ("1", DEBUG_FILL_BYTE, False, False),
        ("2", DEBUG_FILL_BYTE, True, False),
        ("3", DEBUG_FILL_BYTE, True, True),
    ],
)
def test_each_debug_level_adds_to_the_one_below(level, fill_byte, watchdog, device_assert):
    config = DenseConfig.from_env({DENSE_DEBUG_ENV: level})
    assert config.slot_fill_byte == fill_byte
    assert config.slot_watchdog is watchdog
    assert config.slot_watchdog_assert is device_assert
