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
"""The SM reserve MegaMoE is built with (``megamoe_occupancy_sm_reserve``).

MegaMoE caches its occupancy without the SM reserve in the key and sizes its workspaces from it,
so the reserve that reaches MegaMoE is the one of its first occupancy query.  The executor creator
sets the dense DWDP gather's reserve before the model is built and checks this record afterwards
(``test_dwdp_creator.py``).
"""

import pytest

from tensorrt_llm._torch.cute_dsl_utils import sm_reserve
from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as megamoe_op

pytestmark = pytest.mark.cpu_only

#: SMs of the device the fake ``HardwareInfo`` describes.
_SMS = 148


class _HardwareInfo:
    """``cutlass.utils.HardwareInfo``, with the reserve applied as ``cute_dsl_utils`` does."""

    def get_max_active_clusters(self, cluster_size: int) -> int:
        return (_SMS - sm_reserve()) // cluster_size


@pytest.fixture
def fresh_megamoe_occupancy(monkeypatch):
    """No MegaMoE occupancy query yet, on the fake ``HardwareInfo``; restored after the test."""
    cutlass_utils = pytest.importorskip("cutlass.utils")
    monkeypatch.setattr(cutlass_utils, "HardwareInfo", _HardwareInfo)
    monkeypatch.setattr(megamoe_op, "_MEGAMOE_OCCUPANCY_SM_RESERVE", None)
    megamoe_op._max_active_clusters.cache_clear()
    yield
    megamoe_op._max_active_clusters.cache_clear()


def test_megamoe_keeps_the_occupancy_and_the_reserve_of_its_first_query(
    monkeypatch, fresh_megamoe_occupancy
):
    assert megamoe_op.megamoe_occupancy_sm_reserve() is None

    monkeypatch.setenv("TRTLLM_CUTEDSL_SM_RESERVE", "30")
    first = megamoe_op._max_active_clusters(4)
    monkeypatch.setenv("TRTLLM_CUTEDSL_SM_RESERVE", "46")
    megamoe_op._max_active_clusters(2)  # a new cluster size queries again

    assert megamoe_op._max_active_clusters(4) == first == (_SMS - 30) // 4
    assert megamoe_op.megamoe_occupancy_sm_reserve() == 30
