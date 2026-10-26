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
"""The dense DWDP replay guard, a pre-replay hook of ``CUDAGraphRunner``.

A graph replays the dense DWDP schedule captured in it, so right before the replay the dense
manager restores the boundary state (``DenseDwdpManager.before_replay``).  The model registers the
guard when the decoder runner creates its CUDA graph runner
(``KimiLinearForCausalLM.register_cuda_graph_pre_replay_hooks``, forwarded by
``KimiK3ForConditionalGeneration``), through ``DenseDwdpManager.register_replay_guard`` and
``CUDAGraphRunner.register_pre_replay_hook``; only a graph-mode manager registers it.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_kimi_k3_vl import KimiK3ForConditionalGeneration
from tensorrt_llm._torch.models.modeling_kimi_linear import KimiLinearForCausalLM
from tensorrt_llm._torch.modules.dwdp.dense.manager import DenseDwdpManager
from tensorrt_llm._torch.modules.dwdp.dense.types import DenseMode, ForwardState
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner, KeyType
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_KEY = KeyType(batch_size=2, draft_len=0)


def _runner(log: list) -> CUDAGraphRunner:
    """A runner assembled without ``__init__``, with one graph for ``_KEY`` that logs its replay
    and the static input ids it replays with."""
    runner = object.__new__(CUDAGraphRunner)
    runner.config = SimpleNamespace(use_mrope=False)
    runner.max_beam_width = 1
    runner._pre_replay_hooks = []
    runner._pre_replay_position_ids = None
    runner.graph_metadata = {_KEY: {"attn_metadata": None, "spec_metadata": None}}
    runner.shared_static_tensors = {
        "input_ids": torch.zeros(4, dtype=torch.int32),
        "position_ids": torch.zeros((1, 4), dtype=torch.int32),
    }
    static_ids = runner.shared_static_tensors["input_ids"]
    graph = SimpleNamespace(
        replay=lambda: log.append(("graph.replay", static_ids[:2].tolist())), reset=lambda: None
    )
    runner.graphs = {_KEY: graph}
    runner.graph_outputs = {_KEY: "outputs"}
    runner.memory_pool = None
    return runner


def _replay(runner: CUDAGraphRunner) -> object:
    inputs = {
        "attn_metadata": None,
        "input_ids": torch.tensor([7, 9], dtype=torch.int32),
        "position_ids": torch.ones((1, 2), dtype=torch.int32),
    }
    return runner.replay(_KEY, inputs)


def _manager(
    graph_mode: bool, log: list, state: ForwardState, runner: CUDAGraphRunner
) -> DenseDwdpManager:
    """A standalone manager whose runtime is a stand-in in ``state`` that logs its recovery, with
    the static input ids of ``runner`` at that time."""
    mapping = Mapping(world_size=4, tp_size=4, rank=0, moe_ep_size=4, enable_attention_dp=True)
    manager = DenseDwdpManager(DenseMode.STANDALONE, mapping=mapping, graph_mode=graph_mode)
    static_ids = runner.shared_static_tensors["input_ids"]
    manager._dense = SimpleNamespace(
        forward_state=state, recover=lambda: log.append(("recover", static_ids[:2].tolist()))
    )
    return manager


def _register(model: str, manager: DenseDwdpManager | None, runner: CUDAGraphRunner) -> None:
    """``register_cuda_graph_pre_replay_hooks`` of the causal LM or of the multimodal wrapper,
    on stand-ins that hold what the methods read."""
    causal_lm = SimpleNamespace(model=SimpleNamespace(_dense_dwdp=manager))
    if model == "causal_lm":
        KimiLinearForCausalLM.register_cuda_graph_pre_replay_hooks(causal_lm, runner)
        return
    causal_lm.register_cuda_graph_pre_replay_hooks = (
        lambda hooked: KimiLinearForCausalLM.register_cuda_graph_pre_replay_hooks(causal_lm, hooked)
    )
    KimiK3ForConditionalGeneration.register_cuda_graph_pre_replay_hooks(
        SimpleNamespace(llm=causal_lm), runner
    )


@pytest.mark.parametrize("model", ["causal_lm", "multimodal"])
@pytest.mark.parametrize(
    "case, expected",
    [
        ("broken", [("recover", [7, 9]), ("graph.replay", [7, 9])]),
        ("idle", [("graph.replay", [7, 9])]),
        ("eager_manager", [("graph.replay", [7, 9])]),
        ("no_manager", [("graph.replay", [7, 9])]),
    ],
)
def test_the_model_registers_the_replay_guard_of_a_graph_mode_manager(model, case, expected):
    """In graph mode a replay off the forward boundary recovers first, once the static inputs
    hold the replay's input ids; an idle runtime replays as is; without a graph-mode manager there
    is no hook."""
    log = []
    state = ForwardState.BROKEN if case == "broken" else ForwardState.IDLE
    runner = _runner(log)
    manager = (
        None if case == "no_manager" else _manager(case != "eager_manager", log, state, runner)
    )

    _register(model, manager, runner)

    assert len(runner._pre_replay_hooks) == (case in ("broken", "idle"))
    assert _replay(runner) == "outputs"
    assert log == expected


def test_the_guard_looks_up_before_replay_at_every_replay():
    """The hook calls the manager's ``before_replay`` as it is when the graph replays."""
    log = []
    runner = _runner(log)
    manager = _manager(True, log, ForwardState.IDLE, runner)
    assert manager.register_replay_guard(runner)
    manager.before_replay = lambda: log.append("before_replay")

    _replay(runner)
    _replay(runner)

    assert log == ["before_replay", ("graph.replay", [7, 9])] * 2


def test_a_released_manager_guards_nothing():
    """After ``cleanup`` the manager has no runtime, so its registered guard does nothing."""
    log = []
    runner = _runner(log)
    manager = _manager(True, log, ForwardState.BROKEN, runner)
    assert manager.register_replay_guard(runner)
    manager._dense = None

    _replay(runner)

    assert log == [("graph.replay", [7, 9])]
