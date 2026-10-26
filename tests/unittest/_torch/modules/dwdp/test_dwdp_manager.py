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
"""Unit tests for the DwdpManager facade.

DwdpManager is a thin lifecycle layer that:
- creates a DWDP MPI sub-communicator
- owns the Single Source of Truth for registered MoE layer indices
- forwards runtime calls to the DWDPWeightManager produced by setup_dwdp
- creates the attached DenseDwdpManager and registers it with itself, sets
  dense DWDP up after setup_dwdp and hands the runtime and the weight manager
  to it, so the dense manager primes the dense weights together with the
  expert prefetch; releases it before the expert teardown

These tests mock out COMM_WORLD / dwdp_global_rank / setup_dwdp /
setup_dense_dwdp so the full lifecycle can be exercised on a single CPU without
MPI / CUDA / MNNVL. Dense DWDP is off unless a test turns it on.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn

from tensorrt_llm._torch.distributed import MPIDist
from tensorrt_llm._torch.modules.dwdp.dense.config import DENSE_ENV
from tensorrt_llm._torch.modules.dwdp.dense.manager import (
    DenseDwdpManager,
    get_global_dense_dwdp_manager,
    set_global_dense_dwdp_manager,
)
from tensorrt_llm._torch.modules.dwdp.dense.types import DenseMode, DenseSetupError
from tensorrt_llm._torch.pyexecutor.dwdp import (
    DwdpManager,
    get_global_dwdp_manager,
    set_global_dwdp_manager,
)
from tensorrt_llm.llmapi.llm_args import DwdpConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_FAC = "tensorrt_llm._torch.pyexecutor.dwdp"
_DENSE_SETUP = "tensorrt_llm._torch.modules.dwdp.dense.setup"


def _make_config(dwdp_size: int = 2, num_groups: int = 1) -> DwdpConfig:
    return DwdpConfig(
        dwdp_size=dwdp_size,
        num_groups=num_groups,
        num_experts_per_worker=4,
        num_prefetch_experts=4,
    )


def _make_mapping(dwdp_size: int = 2, dwdp_rank: int = 0) -> Mapping:
    return Mapping(
        world_size=dwdp_size,
        rank=dwdp_rank,
        tp_size=dwdp_size,
        dwdp_size=dwdp_size,
        dwdp_rank=dwdp_rank,
    )


def _configure_mock_comm_world(mock_comm_world, world_size: int = 1024):
    """Configure a class-decorator-patched COMM_WORLD mock with sensible defaults.

    Tests that construct a real DwdpManager need both ``Create_group`` (returns
    the dwdp sub-comm) and ``Get_size`` (used by num_groups validation) to be
    set; raw MagicMock returns from ``Get_size`` would fail integer comparison.
    """
    sub_comm = MagicMock()
    mock_comm_world.Create_group.return_value = sub_comm
    mock_comm_world.group.Incl.return_value = MagicMock()
    mock_comm_world.Get_size.return_value = world_size
    return sub_comm


def _start(test: unittest.TestCase, patcher):
    """Start ``patcher`` for the rest of ``test`` and return its mock."""
    mock = patcher.start()
    test.addCleanup(patcher.stop)
    return mock


def _set_dense_env(test: unittest.TestCase, value: str | None) -> None:
    """Set ``TRTLLM_DWDP_DENSE`` (None: unset) for the rest of ``test``."""
    _start(test, patch.dict(os.environ))
    os.environ.pop(DENSE_ENV, None)
    if value is not None:
        os.environ[DENSE_ENV] = value


def _clear_global_managers() -> None:
    """Unregister the DWDP and the dense DWDP manager; a leftover one fails the next ``__init__``."""
    set_global_dwdp_manager(None)
    set_global_dense_dwdp_manager(None)


@patch("tensorrt_llm._torch.pyexecutor.dwdp.setup_dwdp")
@patch("tensorrt_llm._torch.pyexecutor.dwdp.dwdp_global_rank", return_value=0)
@patch("tensorrt_llm._torch.pyexecutor.dwdp.COMM_WORLD")
class TestDwdpManagerLifecycle(unittest.TestCase):
    def setUp(self):
        # Ensure no leftover global singleton from a prior test.
        _clear_global_managers()
        # Dense DWDP off: a model without dense roots, TRTLLM_DWDP_DENSE unset.
        _set_dense_env(self, None)
        _start(self, patch(f"{_FAC}.dense_roots_available", return_value=False))
        self.setup_dense = _start(self, patch(f"{_FAC}.setup_dense_dwdp"))

    def tearDown(self):
        _clear_global_managers()
        self.setup_dense.assert_not_called()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def test_init_rejects_existing_singleton_before_creating_comm(
        self, mock_comm_world, _rank, _setup
    ):
        set_global_dwdp_manager(MagicMock())

        with self.assertRaisesRegex(RuntimeError, "already registered globally"):
            DwdpManager(
                config=_make_config(),
                dist=MagicMock(spec=MPIDist),
                mapping=_make_mapping(),
            )

        mock_comm_world.Create_group.assert_not_called()

    def test_init_rejects_a_registered_dense_manager_before_creating_comm(
        self, mock_comm_world, _rank, _setup
    ):
        """``__enter__`` would fail to register the attached dense manager: refuse before the group."""
        _configure_mock_comm_world(mock_comm_world)
        stray = DenseDwdpManager(DenseMode.ATTACHED).__enter__()

        with self.assertRaisesRegex(RuntimeError, "^DenseDwdpManager already registered globally$"):
            DwdpManager(
                config=_make_config(),
                dist=MagicMock(spec=MPIDist),
                mapping=_make_mapping(),
            )

        mock_comm_world.Create_group.assert_not_called()
        self.assertIsNone(get_global_dwdp_manager())
        self.assertIs(get_global_dense_dwdp_manager(), stray)

    def test_init_creates_an_inert_attached_dense_manager(self, mock_comm_world, _rank, _setup):
        """Created whether or not dense DWDP is requested; registered only by ``__enter__``."""
        _configure_mock_comm_world(mock_comm_world)
        for value in (None, "1"):
            with self.subTest(TRTLLM_DWDP_DENSE=value):
                _set_dense_env(self, value)
                mgr = DwdpManager(
                    config=_make_config(),
                    dist=MagicMock(spec=MPIDist),
                    mapping=_make_mapping(),
                )

                self.assertIsInstance(mgr.dense_manager, DenseDwdpManager)
                self.assertIs(mgr.dense_manager.mode, DenseMode.ATTACHED)
                self.assertIsNone(mgr.dense_manager.dense)
                self.assertIsNone(mgr.dense)
                self.assertIsNone(get_global_dense_dwdp_manager())
                with self.assertRaises(AttributeError):
                    mgr.dense_manager = None

    def test_init_stores_fields(self, mock_comm_world, _rank, _setup):
        sub_comm = MagicMock()
        mock_comm_world.Create_group.return_value = sub_comm
        mock_comm_world.group.Incl.return_value = MagicMock()
        mock_comm_world.Get_size.return_value = 1024

        mgr = DwdpManager(
            config=_make_config(dwdp_size=2),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(dwdp_size=2, dwdp_rank=0),
        )
        self.assertEqual(mgr.dwdp_size, 2)
        self.assertEqual(mgr.dwdp_rank, 0)
        self.assertEqual(mgr._registered_layers, [])
        self.assertIs(mgr.dwdp_group, sub_comm)
        self.assertIsNone(mgr._weight_manager)
        # start_expert_id / end_expert_id derive from
        # num_prefetch_experts (stride) and num_experts_per_worker (size).
        # rank=0: start=0, end=4.
        self.assertEqual(mgr.start_expert_id, 0)
        self.assertEqual(mgr.end_expert_id, 4)

    def test_init_expert_range_uniform(self, mock_comm_world, _rank, _setup):
        """Uniform partition (size == stride): each rank's range is a clean slice."""
        _configure_mock_comm_world(mock_comm_world)
        # dwdp_rank=1 with stride=4 -> start=4, end=4+4=8.
        mgr = DwdpManager(
            config=_make_config(dwdp_size=2),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(dwdp_size=2, dwdp_rank=1),
        )
        self.assertEqual(mgr.start_expert_id, 4)
        self.assertEqual(mgr.end_expert_id, 8)

    def test_init_expert_range_redundancy(self, mock_comm_world, _rank, _setup):
        """Redundancy partition (size > stride): adjacent ranges overlap."""
        _configure_mock_comm_world(mock_comm_world)
        redundant_config = DwdpConfig(
            dwdp_size=2,
            num_groups=1,
            num_experts_per_worker=6,  # storage size
            num_prefetch_experts=4,  # stride < size -> 2-expert overlap
        )
        # rank=0: [0, 6). rank=1 would be [4, 10) -> 2-expert overlap [4, 6).
        mgr = DwdpManager(
            config=redundant_config,
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(dwdp_size=2, dwdp_rank=0),
        )
        self.assertEqual(mgr.start_expert_id, 0)
        self.assertEqual(mgr.end_expert_id, 6)
        # Stored config knobs match what's plumbed into setup_dwdp.
        self.assertEqual(mgr.num_experts_per_worker, 6)
        self.assertEqual(mgr.num_prefetch_experts, 4)

    # ------------------------------------------------------------------
    # num_groups topology validation (was a dead schema field through 1fbc0d49)
    # ------------------------------------------------------------------

    def test_init_accepts_multi_group_topology(self, mock_comm_world, _rank, _setup):
        # num_groups=2, dwdp_size=2 -> 4 CTX ranks total. World >= 4 is fine.
        _configure_mock_comm_world(mock_comm_world, world_size=8)
        mgr = DwdpManager(
            config=DwdpConfig(
                dwdp_size=2,
                num_groups=2,  # valid: this rank (0) is in group 0
                num_experts_per_worker=4,
                num_prefetch_experts=4,
            ),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        self.assertEqual(mgr.num_groups, 2)

    # ------------------------------------------------------------------
    # Global singleton + enter/exit
    # ------------------------------------------------------------------

    def test_enter_registers_global(self, mock_comm_world, _rank, _setup):
        mock_comm_world.Create_group.return_value = MagicMock()
        mock_comm_world.group.Incl.return_value = MagicMock()
        mock_comm_world.Get_size.return_value = 1024
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        self.assertIsNone(get_global_dwdp_manager())
        self.assertIsNone(get_global_dense_dwdp_manager())
        mgr.__enter__()
        self.assertIs(get_global_dwdp_manager(), mgr)
        self.assertIs(get_global_dense_dwdp_manager(), mgr.dense_manager)
        mgr.__exit__(None, None, None)
        self.assertIsNone(get_global_dwdp_manager())
        self.assertIsNone(get_global_dense_dwdp_manager())

    def test_enter_frees_the_group_when_the_dense_manager_cannot_register(
        self, mock_comm_world, _rank, _setup
    ):
        """A dense manager registered after ``__init__``: neither manager stays registered.

        ``__enter__`` frees the DWDP group, unregisters the DWDP manager and re-raises; nothing
        exits it later, and the other dense manager stays registered.
        """
        sub_comm = _configure_mock_comm_world(mock_comm_world)
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        stray = DenseDwdpManager(DenseMode.ATTACHED).__enter__()

        with self.assertRaisesRegex(RuntimeError, "^DenseDwdpManager already registered globally$"):
            mgr.__enter__()

        sub_comm.Free.assert_called_once_with()
        self.assertIsNone(mgr.dwdp_group)
        self.assertIsNone(get_global_dwdp_manager())
        self.assertIs(get_global_dense_dwdp_manager(), stray)
        with self.assertRaisesRegex(RuntimeError, r"setup\(\) called after cleanup\(\)"):
            mgr.setup(MagicMock())

    def test_enter_logs_a_failing_free_and_raises_the_registration_error(
        self, mock_comm_world, _rank, _setup
    ):
        sub_comm = _configure_mock_comm_world(mock_comm_world)
        sub_comm.Free.side_effect = RuntimeError("free failed")
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        set_global_dense_dwdp_manager(MagicMock())

        with patch(f"{_FAC}.logger.error") as log_error:
            with self.assertRaisesRegex(RuntimeError, "DenseDwdpManager already registered"):
                mgr.__enter__()

        sub_comm.Free.assert_called_once_with()
        (message,), _ = log_error.call_args
        self.assertIn("Failed to free the DWDP group after the dense DWDP manager", message)
        self.assertIn("free failed", message)
        self.assertIsNone(get_global_dwdp_manager())

    def test_cleanup_frees_comm_and_idempotent(self, mock_comm_world, _rank, _setup):
        sub_comm = MagicMock()
        mock_comm_world.Create_group.return_value = sub_comm
        mock_comm_world.group.Incl.return_value = MagicMock()
        mock_comm_world.Get_size.return_value = 1024
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        mgr.cleanup()
        sub_comm.Free.assert_called_once()
        self.assertIsNone(mgr.dwdp_group)
        # Idempotent — second call must not raise
        mgr.cleanup()
        sub_comm.Free.assert_called_once()  # still only one Free

    def test_exit_clears_global_even_when_cleanup_fails(self, mock_comm_world, _rank, _setup):
        _configure_mock_comm_world(mock_comm_world)
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        mgr.__enter__()

        with patch.object(mgr, "cleanup", side_effect=RuntimeError("cleanup failed")):
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                mgr.__exit__(None, None, None)

        self.assertIsNone(get_global_dwdp_manager())
        self.assertIsNone(get_global_dense_dwdp_manager())

    def test_cleanup_frees_comm_when_weight_release_fails(self, mock_comm_world, _rank, _setup):
        sub_comm = _configure_mock_comm_world(mock_comm_world)
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        mgr._weight_manager = MagicMock()
        mgr._weight_manager.release.side_effect = RuntimeError("release failed")

        with self.assertRaisesRegex(RuntimeError, "release failed"):
            mgr.cleanup()

        sub_comm.Free.assert_called_once_with()
        self.assertIsNone(mgr.dwdp_group)

    def test_cleanup_logs_a_failing_free_after_a_release_error(
        self, mock_comm_world, _rank, _setup
    ):
        """The release error is raised; the Free failure after it is logged through ``error``."""
        sub_comm = _configure_mock_comm_world(mock_comm_world)
        sub_comm.Free.side_effect = RuntimeError("free failed")
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        mgr._weight_manager = MagicMock()
        mgr._weight_manager.release.side_effect = RuntimeError("release failed")

        # Only ``error`` is replaced: any other logger method the path calls must exist.
        with patch(f"{_FAC}.logger.error") as log_error:
            with self.assertRaisesRegex(RuntimeError, "release failed"):
                mgr.cleanup()

        sub_comm.Free.assert_called_once_with()
        (message,), _ = log_error.call_args
        self.assertIn("Failed to free DWDP group", message)
        self.assertIn("free failed", message)
        self.assertIsNone(mgr.dwdp_group)
        self.assertIsNone(mgr._weight_manager)

    def test_cleanup_releases_weight_manager_once(self, mock_comm_world, _rank, _setup):
        sub_comm = _configure_mock_comm_world(mock_comm_world)
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        weight_manager = MagicMock()
        mgr._weight_manager = weight_manager

        mgr.cleanup()
        mgr.cleanup()

        weight_manager.release.assert_called_once_with()
        sub_comm.Free.assert_called_once_with()

    # ------------------------------------------------------------------
    # add_layer (SSOT)
    # ------------------------------------------------------------------

    def test_add_layer_appends_to_registered(self, mock_comm_world, _rank, _setup):
        mock_comm_world.Create_group.return_value = MagicMock()
        mock_comm_world.group.Incl.return_value = MagicMock()
        mock_comm_world.Get_size.return_value = 1024
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        mgr.add_layer(3)
        mgr.add_layer(5)
        mgr.add_layer(7)
        self.assertEqual(mgr._registered_layers, [3, 5, 7])

    def test_add_layer_duplicate_is_idempotent(self, mock_comm_world, _rank, _setup):
        mock_comm_world.Create_group.return_value = MagicMock()
        mock_comm_world.group.Incl.return_value = MagicMock()
        mock_comm_world.Get_size.return_value = 1024
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        mgr.add_layer(3)
        mgr.add_layer(3)
        self.assertEqual(mgr._registered_layers, [3])

    # ------------------------------------------------------------------
    # setup(model) forwards to setup_dwdp with layer_indices SSOT
    # ------------------------------------------------------------------

    def test_setup_forwards_layer_indices(self, mock_comm_world, _rank, mock_setup):
        sub_comm = MagicMock()
        mock_comm_world.Create_group.return_value = sub_comm
        mock_comm_world.group.Incl.return_value = MagicMock()
        mock_comm_world.Get_size.return_value = 1024
        fake_weight_manager = MagicMock()
        mock_setup.return_value = fake_weight_manager

        mapping = _make_mapping()
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=mapping,
        )
        # Registration order should be sorted before passing to setup_dwdp
        mgr.add_layer(7)
        mgr.add_layer(3)
        mgr.add_layer(5)

        fake_model = MagicMock()
        with patch(
            "tensorrt_llm._torch.pyexecutor.dwdp.torch.cuda.current_device",
            return_value=0,
        ):
            result = mgr.setup(fake_model)

        mock_setup.assert_called_once()
        _, kwargs = mock_setup.call_args
        self.assertIs(kwargs["model"], fake_model)
        self.assertIs(kwargs["mapping"], mapping)
        self.assertEqual(kwargs["device_id"], 0)
        self.assertIs(kwargs["comm"], sub_comm)
        self.assertEqual(kwargs["layer_indices"], [3, 5, 7])  # sorted SSOT
        # Config-driven expert range fields are forwarded from DwdpConfig.
        self.assertEqual(kwargs["num_experts_per_worker"], 4)
        self.assertEqual(kwargs["num_prefetch_experts"], 4)
        self.assertIs(result, fake_weight_manager)
        self.assertIs(mgr._weight_manager, fake_weight_manager)

    # ------------------------------------------------------------------
    # Runtime forwards require setup()
    # ------------------------------------------------------------------

    def test_prefetch_first_layers_forwards(self, mock_comm_world, _rank, mock_setup):
        mock_comm_world.Create_group.return_value = MagicMock()
        mock_comm_world.group.Incl.return_value = MagicMock()
        mock_comm_world.Get_size.return_value = 1024
        fake_wm = MagicMock()
        fake_wm.first_moe_layer.return_value = 3
        fake_wm.next_moe_layer.return_value = 5
        mock_setup.return_value = fake_wm
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        with patch(
            "tensorrt_llm._torch.pyexecutor.dwdp.torch.cuda.current_device",
            return_value=0,
        ):
            mgr.setup(MagicMock())
        mgr.prefetch_first_layers()
        # First MoE layer 3 + ping-pong depth -> also prefetch next (5)
        self.assertEqual(
            fake_wm.prefetch_layer.call_args_list,
            [
                unittest.mock.call(3),
                unittest.mock.call(5),
            ],
        )

    def test_record_compute_and_prefetch_next_schedules_next_next(
        self, mock_comm_world, _rank, mock_setup
    ):
        mock_comm_world.Create_group.return_value = MagicMock()
        mock_comm_world.group.Incl.return_value = MagicMock()
        mock_comm_world.Get_size.return_value = 1024
        fake_wm = MagicMock()
        fake_wm.next_moe_layer.side_effect = lambda layer_idx: {7: 9, 9: 11}.get(layer_idx)
        mock_setup.return_value = fake_wm
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        with patch(
            "tensorrt_llm._torch.pyexecutor.dwdp.torch.cuda.current_device",
            return_value=0,
        ):
            mgr.setup(MagicMock())

        mgr.record_compute_and_prefetch_next(7)
        fake_wm.record_compute.assert_called_once_with(7)
        self.assertEqual(
            fake_wm.next_moe_layer.call_args_list,
            [unittest.mock.call(7), unittest.mock.call(9)],
        )
        fake_wm.prefetch_layer.assert_called_once_with(11)

    def test_record_compute_and_prefetch_next_last_layer_is_noop(
        self, mock_comm_world, _rank, mock_setup
    ):
        mock_comm_world.Create_group.return_value = MagicMock()
        mock_comm_world.group.Incl.return_value = MagicMock()
        mock_comm_world.Get_size.return_value = 1024
        fake_wm = MagicMock()
        fake_wm.next_moe_layer.return_value = None
        mock_setup.return_value = fake_wm
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        with patch(
            "tensorrt_llm._torch.pyexecutor.dwdp.torch.cuda.current_device",
            return_value=0,
        ):
            mgr.setup(MagicMock())

        mgr.record_compute_and_prefetch_next(60)
        fake_wm.record_compute.assert_called_once_with(60)
        fake_wm.prefetch_layer.assert_not_called()

    def test_prefetch_pipeline_schedules_each_layer_once(self, mock_comm_world, _rank, mock_setup):
        mock_comm_world.Create_group.return_value = MagicMock()
        mock_comm_world.group.Incl.return_value = MagicMock()
        mock_comm_world.Get_size.return_value = 1024
        fake_wm = MagicMock()
        fake_wm.first_moe_layer.return_value = 3
        fake_wm.next_moe_layer.side_effect = lambda layer_idx: {
            3: 5,
            5: 7,
            7: 9,
            9: None,
        }.get(layer_idx)
        mock_setup.return_value = fake_wm
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        with patch(
            "tensorrt_llm._torch.pyexecutor.dwdp.torch.cuda.current_device",
            return_value=0,
        ):
            mgr.setup(MagicMock())

        mgr.prefetch_first_layers()
        for layer_idx in (3, 5, 7, 9):
            mgr.record_compute_and_prefetch_next(layer_idx)

        self.assertEqual(
            fake_wm.prefetch_layer.call_args_list,
            [
                unittest.mock.call(3),
                unittest.mock.call(5),
                unittest.mock.call(7),
                unittest.mock.call(9),
            ],
        )

    def test_wait_and_bind_forwards(self, mock_comm_world, _rank, mock_setup):
        mock_comm_world.Create_group.return_value = MagicMock()
        mock_comm_world.group.Incl.return_value = MagicMock()
        mock_comm_world.Get_size.return_value = 1024
        fake_wm = MagicMock()
        mock_setup.return_value = fake_wm
        mgr = DwdpManager(
            config=_make_config(),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(),
        )
        with patch(
            "tensorrt_llm._torch.pyexecutor.dwdp.torch.cuda.current_device",
            return_value=0,
        ):
            mgr.setup(MagicMock())
        backend = MagicMock()
        mgr.wait_and_bind(backend, 5)
        fake_wm.wait_and_bind.assert_called_once_with(backend, 5)


# ----------------------------------------------------------------------
# Dense DWDP wiring
# ----------------------------------------------------------------------


class _FakeWeightManager:
    """The DWDPWeightManager calls DwdpManager and its dense manager make, appended to a call log."""

    def __init__(self, log: list[tuple], layers: list[int]) -> None:
        self.log = log
        self.layers = list(layers)

    def first_moe_layer(self) -> int:
        return self.layers[0]

    def next_moe_layer(self, layer: int) -> int | None:
        return next((later for later in self.layers if later > layer), None)

    def prefetch_layer(self, layer: int) -> None:
        self.log.append(("E", layer))

    def record_compute(self, layer: int) -> None:
        self.log.append(("record_compute", layer))

    def wait_and_bind(self, backend: object, layer: int) -> None:
        self.log.append(("wait_and_bind", layer))

    def release(self) -> None:
        self.log.append(("wm.release",))


class _FakeDense:
    """The DenseDWDP calls the attached dense manager makes, appended to the shared call log.

    The flags follow the runtime: ``war_barrier`` clears ``primed`` and
    ``forward_open``, ``mark_primed`` sets ``primed``.
    """

    def __init__(
        self,
        log: list[tuple],
        *,
        primed: bool = False,
        forward_open: bool = False,
        release_error: BaseException | None = None,
        eager_error: BaseException | None = None,
    ) -> None:
        self.log = log
        self.primed = primed
        self.forward_open = forward_open
        self.release_error = release_error
        self.eager_error = eager_error
        self.eager_checks: list[str] = []

    def require_eager(self, hook: str) -> None:
        self.eager_checks.append(hook)
        if self.eager_error is not None:
            raise self.eager_error

    def prefetch(self, layer: int) -> None:
        self.log.append(("L", layer))

    def mark_primed(self) -> None:
        self.log.append(("mark_primed",))
        self.primed = True

    def war_barrier(self) -> None:
        self.log.append(("war_barrier",))
        self.primed = False
        self.forward_open = False

    def release(self) -> None:
        self.log.append(("dense.release",))
        if self.release_error is not None:
            raise self.release_error


class _DenseModel(nn.Module):
    """A model the fake dense setup marks the way the real one does."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()


class _DenseManagerTest(unittest.TestCase):
    """A DwdpManager over fakes: one call log across dense, weight manager and teardown.

    The manager is not entered unless a test enters it; its dense manager is real.
    """

    LAYERS = [3, 5, 7]
    #: Global rank 3 is DWDP rank 1 of group 1 and drives CUDA device 2: three different
    #: numbers, so passing one in place of another shows.
    GLOBAL_RANK = 3
    DWDP_RANK = 1
    DEVICE_ID = 2

    def setUp(self):
        _clear_global_managers()
        self.addCleanup(_clear_global_managers)
        self.log: list[tuple] = []
        sub_comm = _configure_mock_comm_world(_start(self, patch(f"{_FAC}.COMM_WORLD")))
        sub_comm.Free.side_effect = lambda: self.log.append(("group.Free",))
        self.sub_comm = sub_comm
        _start(self, patch(f"{_FAC}.dwdp_global_rank", return_value=self.GLOBAL_RANK))
        _start(self, patch(f"{_FAC}.torch.cuda.current_device", return_value=self.DEVICE_ID))
        self.weight_manager = _FakeWeightManager(self.log, self.LAYERS)
        self.setup_dwdp = _start(
            self, patch(f"{_FAC}.setup_dwdp", return_value=self.weight_manager)
        )
        self.teardown = _start(
            self,
            patch(
                f"{_FAC}.teardown_dwdp",
                side_effect=lambda model, expected_manager: self.log.append(("teardown_dwdp",)),
            ),
        )
        self.roots_available = _start(
            self, patch(f"{_FAC}.dense_roots_available", return_value=True)
        )
        self.setup_dense = _start(self, patch(f"{_FAC}.setup_dense_dwdp", return_value=None))
        _set_dense_env(self, None)
        self.model = _DenseModel()
        self.manager = DwdpManager(
            config=_make_config(num_groups=2),
            dist=MagicMock(spec=MPIDist),
            mapping=_make_mapping(dwdp_rank=self.DWDP_RANK),
        )
        for layer in reversed(self.LAYERS):
            self.manager.add_layer(layer)

    def setup_with_dense(self, dense: object) -> None:
        """Run ``DwdpManager.setup`` with a dense setup that commits ``dense``."""

        def commit(model, *_args, **_kwargs):
            model._dense_dwdp_active = True
            model.model._dense_dwdp_active = True
            return dense

        self.setup_dense.side_effect = commit
        self.manager.setup(self.model)


class TestDwdpManagerDenseSetup(_DenseManagerTest):
    def test_setup_runs_dense_setup_when_roots_are_available_but_not_requested(self):
        """Collective #1 runs on every rank of a model with roots, requested or not."""
        seen = {}

        def dense_setup(model, weight_manager, comm, **kwargs):
            seen["model_ref"] = self.manager._model_ref()
            seen["weight_manager"] = self.manager._weight_manager
            return None

        self.setup_dense.side_effect = dense_setup
        attach = _start(self, patch.object(self.manager.dense_manager, "attach"))

        result = self.manager.setup(self.model)

        self.assertEqual(self.setup_dwdp.call_args.kwargs["device_id"], self.DEVICE_ID)
        self.roots_available.assert_called_once_with(self.model)
        self.setup_dense.assert_called_once_with(
            self.model,
            self.weight_manager,
            self.sub_comm,
            dwdp_rank=self.DWDP_RANK,
            dwdp_size=2,
            device_id=self.DEVICE_ID,
            moe_layers=[3, 5, 7],
        )
        # It runs after the expert setup and after the model reference is kept.
        self.assertIs(seen["model_ref"], self.model)
        self.assertIs(seen["weight_manager"], self.weight_manager)
        self.assertIs(result, self.weight_manager)
        self.assertIsNone(self.manager.dense)
        attach.assert_not_called()

    def test_setup_hands_the_committed_runtime_and_the_weight_manager_to_the_dense_manager(self):
        dense = _FakeDense(self.log)
        dense_manager = self.manager.dense_manager
        attach = _start(self, patch.object(dense_manager, "attach", wraps=dense_manager.attach))

        self.setup_with_dense(dense)

        attach.assert_called_once_with(dense, expert=self.weight_manager)
        self.assertIs(dense_manager.dense, dense)
        self.assertIs(self.manager.dense, dense)
        self.assertEqual(self.log, [])

    def _request_dense_without_roots(self, **switches: str) -> None:
        """A model without roots; ``TRTLLM_DWDP_DENSE=1`` plus ``switches`` (name suffixes)."""
        self.roots_available.return_value = False
        _set_dense_env(self, "1")
        for name in [name for name in os.environ if name.startswith(f"{DENSE_ENV}_")]:
            del os.environ[name]
        for suffix, value in switches.items():
            os.environ[f"{DENSE_ENV}_{suffix}"] = value

    def test_a_rank_requesting_dense_without_roots_raises_alone_and_at_once(self):
        """No rank makes a dense collective on a model without roots, so none waits for one.

        A requesting rank refuses by itself right after the expert setup: it raises unless its
        own ``TRTLLM_DWDP_DENSE_REQUIRED`` is off, and switches it cannot read count as required.
        """
        cases = ({}, {"REQUIRED": "1"}, {"REQUIRED": "maybe"}, {"REQUIRED": "0", "SEGMENTS": "atn"})
        for switches in cases:
            with self.subTest(**switches):
                self._request_dense_without_roots(**switches)
                with self.assertRaisesRegex(
                    DenseSetupError,
                    "TRTLLM_DWDP_DENSE asks for dense DWDP, but the decoder layers of _DenseModel "
                    "do not provide dense_dwdp_roots",
                ):
                    self.manager.setup(self.model)
                self.assertIsNone(self.manager.dense)
        self.assertEqual(self.setup_dwdp.call_count, len(cases))
        self.setup_dense.assert_not_called()
        self.sub_comm.allgather.assert_not_called()

        self.manager.cleanup()

        self.assertEqual(self.log, [("teardown_dwdp",), ("group.Free",)])

    def test_a_rank_requesting_dense_without_roots_with_required_off_keeps_serving(self):
        self._request_dense_without_roots(REQUIRED="0")
        log_error = _start(self, patch(f"{_DENSE_SETUP}.logger.error"))

        result = self.manager.setup(self.model)

        self.assertIs(result, self.weight_manager)
        self.assertIsNone(self.manager.dense)
        self.setup_dense.assert_not_called()
        self.sub_comm.allgather.assert_not_called()
        (message,), _ = log_error.call_args
        self.assertIn("do not provide dense_dwdp_roots", message)
        self.assertIn("TRTLLM_DWDP_DENSE_REQUIRED=0, so dense DWDP is off on this rank", message)
        self.manager.prefetch_first_layers()
        self.assertEqual(self.log, [("E", 3), ("E", 5)])

    def test_setup_skips_dense_without_roots_or_request(self):
        """DeepSeek path: no dense setup, no collective, expert-only priming."""
        self.roots_available.return_value = False
        for value in (None, "0"):
            with self.subTest(TRTLLM_DWDP_DENSE=value):
                _set_dense_env(self, value)
                self.manager.setup(self.model)
                self.setup_dense.assert_not_called()
                self.assertIsNone(self.manager.dense)

        self.manager.prefetch_first_layers()

        self.assertEqual(self.log, [("E", 3), ("E", 5)])

    def test_dense_setup_error_propagates_and_cleanup_still_tears_down(self):
        self.setup_dense.side_effect = DenseSetupError("dense setup abandoned")

        with self.assertRaisesRegex(DenseSetupError, "dense setup abandoned"):
            self.manager.setup(self.model)
        self.assertIsNone(self.manager.dense)

        self.manager.cleanup()

        self.assertEqual(self.log, [("teardown_dwdp",), ("group.Free",)])


if __name__ == "__main__":
    unittest.main()
