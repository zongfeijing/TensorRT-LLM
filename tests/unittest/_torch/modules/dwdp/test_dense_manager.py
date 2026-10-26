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
"""CPU tests of the dense DWDP manager (``manager.DenseDwdpManager``).

* ``standalone_group`` (with a tag of its own).
"""

from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.modules.dwdp.dense import manager as dense_manager
from tensorrt_llm._torch.modules.dwdp.dense.manager import (
    STANDALONE_GROUP_TAG,
    set_global_dense_dwdp_manager,
    standalone_group,
)
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _no_dense_manager():
    """No test starts or ends with a dense DWDP manager registered."""
    set_global_dense_dwdp_manager(None)
    yield
    set_global_dense_dwdp_manager(None)


def adp_mapping(size, rank=0):
    """Rank ``rank`` of an attention-DP instance of ``size`` ranks with experts parallel (DEP)."""
    return Mapping(
        world_size=size, rank=rank, tp_size=size, moe_ep_size=size, enable_attention_dp=True
    )


def calls(log):
    """``log`` without the event records."""
    return [entry for entry in log if entry != "record"]


# ---------------------------------------------------------------------------
# standalone_group and StandaloneTransfer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mapping, tp_group",
    [
        (adp_mapping(4, rank=2), [0, 1, 2, 3]),
        (
            Mapping(world_size=8, rank=5, tp_size=4, pp_size=2, enable_attention_dp=True),
            [4, 5, 6, 7],
        ),
    ],
    ids=["tp4", "second_pp_stage"],
)
def test_standalone_group_is_a_sub_communicator_over_the_tp_group(monkeypatch, mapping, tp_group):
    """The group is created with a tag of its own, not the default tag 0 of ``MPIDist.tp_comm``.

    ``MPIDist.tp_comm`` creates a group over the same ranks of the same parent, and MPI pairs the
    ``Create_group`` calls on one parent by their tag.
    """
    world = MagicMock()
    monkeypatch.setattr(dense_manager, "mpi_disabled", lambda: False)
    monkeypatch.setattr(dense_manager, "mpi_comm", lambda: world)

    group = standalone_group(mapping)

    assert mapping.tp_group == tp_group
    world.group.Incl.assert_called_once_with(tp_group)
    world.Create_group.assert_called_once_with(
        world.group.Incl.return_value, tag=STANDALONE_GROUP_TAG
    )
    assert group is world.Create_group.return_value
    # Not the default tag, and a valid tag under every MPI (MPI_TAG_UB >= 32767).
    assert 0 < STANDALONE_GROUP_TAG <= 32767
