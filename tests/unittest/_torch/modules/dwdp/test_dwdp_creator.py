# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""PyExecutor construction tests for DWDP: manager ownership and the dense DWDP mode.

The dense DWDP gate's own rules and texts are tested in ``test_dense_gate.py``.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm._torch.modules.dwdp.dense.config import DENSE_ENV
from tensorrt_llm._torch.modules.dwdp.dense.manager import set_global_dense_dwdp_manager
from tensorrt_llm._torch.pyexecutor import py_executor_creator
from tensorrt_llm._torch.pyexecutor.dwdp import get_global_dwdp_manager, set_global_dwdp_manager

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _no_global_managers():
    """Start and end every test without a registered DWDP or dense DWDP manager.

    A leftover registration would make the next ``DwdpManager`` or ``DenseDwdpManager`` of this
    process fail.
    """
    set_global_dwdp_manager(None)
    set_global_dense_dwdp_manager(None)
    yield
    set_global_dwdp_manager(None)
    set_global_dense_dwdp_manager(None)


def test_create_py_executor_rolls_back_new_dwdp_manager_on_error():
    manager = MagicMock()
    manager.__exit__.side_effect = lambda *_args: set_global_dwdp_manager(None)

    def fail_after_registration(**_kwargs):
        set_global_dwdp_manager(manager)
        raise RuntimeError("construction failed")

    with patch.object(
        py_executor_creator,
        "_create_py_executor_impl",
        side_effect=fail_after_registration,
    ):
        with pytest.raises(RuntimeError, match="construction failed"):
            py_executor_creator.create_py_executor(llm_args=MagicMock())

    manager.__exit__.assert_called_once_with(None, None, None)
    assert get_global_dwdp_manager() is None


def test_create_py_executor_leaves_a_pre_existing_dwdp_manager_alone():
    """The wrapper only tears down a manager it created, not one it inherited."""
    existing = MagicMock()
    set_global_dwdp_manager(existing)

    def fail_without_registering(**_kwargs):
        raise RuntimeError("construction failed")

    with patch.object(
        py_executor_creator,
        "_create_py_executor_impl",
        side_effect=fail_without_registering,
    ):
        with pytest.raises(RuntimeError, match="construction failed"):
            py_executor_creator.create_py_executor(llm_args=MagicMock())

    existing.__exit__.assert_not_called()
    assert get_global_dwdp_manager() is existing


# ----------------------------------------------------------------------
# Dense DWDP launch gate and mode
# ----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_dense_switches(monkeypatch):
    """Start every test without ``TRTLLM_DWDP_DENSE*`` variables; tests set the ones they need."""
    for name in [name for name in os.environ if name.startswith(DENSE_ENV)]:
        monkeypatch.delenv(name)
