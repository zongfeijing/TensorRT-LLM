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
"""Dense DWDP on a Kimi-K3 generation server, end to end on 4 GPUs: the tokens stay the same.

The LLM API serves Kimi-K3 NVFP4 cut to the checkpoint's first nine layers (KDA layers 0-2, 4-6
and 8, MLA layers 3 and 7) as a DEP16 generation server serves the whole model: TP4 with attention
DP and EP4 on ``MEGAMOE_CUTEDSL``, decode CUDA graphs of 1, 2 and 4 requests with padding, the
overlap scheduler, the KV cache manager v2 sized by the estimation run (without a host tier), an
FP8 KV cache with 64-token pages, and the DSpark drafter with five drafts.  It needs four
Blackwell or Rubin GPUs (SM100, SM103 or SM107) in one NVLink domain.  Both checkpoints come from
``LLM_MODELS_ROOT`` (``Kimi-K3-NVFP4``, and the drafter as ``RadixArk-Kimi-K3-DSpark``, the name
the Kimi-K3 accuracy tests use, or ``Kimi-K3-DSpark``) through directories of links: the target's
links the checkpoint's files, of the weight shards only those that hold the nine layers, and cuts
the layer count and the layer lists of ``config.json``; the drafter's maps its five target layers
into the nine.

Per transfer, the same requests run four times, each on fresh MPI workers: twice with replicated
weights, a determinism pre-check of the comparison, then twice with standalone dense DWDP in graph
mode (the ``CLOSED`` protocol, wrap priming) over the four ranks with ``TRTLLM_DWDP_DENSE_STRICT``:
with ``TRTLLM_DWDP_DENSE_DEBUG=2``, whose slot watchdog synchronizes the host after every step,
and with ``TRTLLM_DWDP_DENSE_DEBUG=0`` as deployed, where the overlap scheduler enqueues a step
while the one before still runs.  The transfers are the TMA SM gather
(``TRTLLM_DWDP_DENSE_COPY=gather``, Rubin's) and one batched copy-engine copy per layer
(``batched``, GB300's).  Only the dense switches differ between the runs of a transfer: with the
gather every run has its compute-side settings (the SM reserve of the gather's blocks, the
``clc_dynamic`` scheduler and non-cooperative routing), with the copy engines none of them, so no
run has an SM reserve; and every run has expandable allocator segments, two accepted draft tokens
forced per step, so that the drafter's tokens are in the outputs, and the fallback tactic of every
tunable op (no autotuning).

The requests are pinned to attention-DP ranks: five to rank 0, three to rank 1, one to rank 2 and
none to rank 3, with different output lengths.  While rank 0 holds five, every rank steps eagerly
(a batch larger than the largest graph); then the ranks replay graphs, the smaller batches padded
to the largest one, and rank 3 replays attention-DP dummy requests.  The executor takes all the
requests in its first fetch, the only one that waits for requests, so that every run steps through
the same batches: a request's tokens depend on the batches it ran in.

Pass: every request's tokens and accepted draft tokens equal the replicated runs', and the KV cache
holds more tokens with the memory dense DWDP frees.  Probes in the workers show that each dense run
took each path above, that steps of the run without the watchdog began while the step before
still ran, that every replay of a dense run, and none of a replicated one, ran behind the replay
guard the model registers with the CUDA graph runner, that on every rank the runtime ran the
transfer asked for (with the copy engines without an SM reserve in any run) and captured its
graphs, and ended without a demand fetch, a fence fallback, a recovery or a slot watchdog
mismatch, after serving steps that replayed graphs.
"""

from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path
from typing import NamedTuple

import cloudpickle
import pytest
import test_dense_k3_decode_gpu as k3_decode
import torch
from mpi4py import MPI
from test_common.grouped_test_utils import submit_sync_per_worker
from test_dense_k3_decode_gpu import _GATHER_ENV, _MODEL_ENV
from test_dense_k3_modules_gpu import _NUM_LAYERS, _require_gather
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.speculative.interface import FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR
from tensorrt_llm.llmapi import CudaGraphConfig, DSparkDecodingConfig, KvCacheConfig, MoeConfig
from tensorrt_llm.llmapi.mpi_session import MpiPoolSession
from tensorrt_llm.scheduling_params import SchedulingParams

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

# MPIPoolExecutor leaks a worker thread on first use; keep CI green.
pytestmark = pytest.mark.threadleak(enabled=False)

# The K3 decode test's fixture: a Blackwell or Rubin GPU with the model environment.
device = k3_decode.device

_NUM_RANKS = 4
_GRAPH_BATCH_SIZES = [1, 2, 4]
#: The checkpoints under ``LLM_MODELS_ROOT``: the target, and the DSpark drafter under the name
#: that the Kimi-K3 accuracy tests use or under a plain one.
_TARGET_CHECKPOINT = "Kimi-K3-NVFP4"
_DRAFTER_CHECKPOINTS = ("RadixArk-Kimi-K3-DSpark", "Kimi-K3-DSpark")
#: The drafter checkpoint's target layers, ``[2, 23, 47, 71, 89]`` of 0..92, scaled into 0..8.
_DRAFTER_TARGET_LAYERS = [0, 2, 4, 6, 8]
#: Output lengths of the requests pinned to each attention-DP rank; rank 3 gets none.
_REQUESTS = {0: (8, 16, 24, 32, 40), 1: (12, 24, 36), 2: (20,)}
_PROMPTS = (
    "The capital of France is",
    "Water is made of hydrogen and",
    "1 + 1 = 2, 2 + 2 = 4, 4 + 4 =",
    "Once upon a time, in a small village by the sea,",
    "The quick brown fox jumps over",
    "List the first five prime numbers:",
    "Translate to German: good morning",
    "def fibonacci(n):",
    "The speed of light in vacuum is",
)

#: Every run's environment: the deployment's model switch, expandable allocator segments, which
#: return the memory dense DWDP frees to the device, and two accepted draft tokens per step (the
#: nine layers accept almost none), so that the drafter's tokens, which read the hidden states of
#: the dense layers, are in the outputs.
_SERVER_ENV = {
    **_MODEL_ENV,
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR: "2",
}

#: Each transfer's settings, in every run of it: with the gather its compute-side settings,
#: which setup applies to the dense runs (``apply_compute_side_defaults``), so the replicated runs
#: set them too; with the copy engines none.
_TRANSFER_ENV = {"gather": _GATHER_ENV, "batched": {}}

#: The variables the probes read on every rank: the gather's settings.
_GATHER_KNOBS = tuple(_GATHER_ENV)


def _env(transfer: str, debug: int | None = None) -> dict[str, str]:
    """A replicated run's environment (``debug`` None) or a dense one's, with ``transfer``."""
    env = {**_SERVER_ENV, **_TRANSFER_ENV[transfer], "TRTLLM_DWDP_DENSE": "0"}
    if debug is not None:
        env.update(
            TRTLLM_DWDP_DENSE="1",
            TRTLLM_DWDP_DENSE_COPY=transfer,
            TRTLLM_DWDP_DENSE_STRICT="1",
            TRTLLM_DWDP_DENSE_DEBUG=str(debug),
        )
    return env


#: The dense runtime's counters the probes read when the runtime is released.
_COUNTERS = (
    "captures",
    "recoveries",
    "demand_fetches",
    "war_fences",
    "watchdog_checks",
    "watchdog_mismatches",
)

_LAYER_KEY = re.compile(r"\.layers\.(\d+)\.")


class _Run(NamedTuple):
    """What one run served, and what its workers recorded."""

    outputs: list[tuple[list[int], tuple[int, int]]]
    """Per request: its tokens and ``(accepted, drafted)`` draft tokens."""
    kv_tokens: int
    """The tokens the KV cache holds."""
    ranks: list[dict]
    """Per rank: what ``_install_probes`` recorded."""


def _linked(source: Path, out: Path, config: dict, skip: frozenset[str] = frozenset()) -> Path:
    """``out``: ``config`` as ``config.json`` and a link to every other file of ``source``."""
    out.mkdir()
    (out / "config.json").write_text(json.dumps(config))
    for entry in source.iterdir():
        if entry.name not in skip | {"config.json"}:
            (out / entry.name).symlink_to(entry)
    return out


def _checkpoints(target: Path, drafter: Path, out: Path) -> tuple[Path, Path]:
    """The target cut to ``_NUM_LAYERS`` layers, and the drafter remapped onto them."""
    config = json.loads((target / "config.json").read_text())
    text = config["text_config"]
    text["num_hidden_layers"] = _NUM_LAYERS
    for name in ("kda_layers", "full_attn_layers"):  # 1-indexed
        layers = text["linear_attn_config"][name]
        text["linear_attn_config"][name] = [layer for layer in layers if layer <= _NUM_LAYERS]
    weight_map = json.loads((target / "model.safetensors.index.json").read_text())["weight_map"]
    used = {
        shard
        for key, shard in weight_map.items()
        if (match := _LAYER_KEY.search(key)) is None or int(match[1]) < _NUM_LAYERS
    }
    unused = frozenset(weight_map.values()) - used
    draft_config = json.loads((drafter / "config.json").read_text())
    draft_config.update(
        target_layer_ids=_DRAFTER_TARGET_LAYERS, target_num_hidden_layers=_NUM_LAYERS
    )
    return (
        _linked(target, out / "target", config, unused),
        _linked(drafter, out / "drafter", draft_config),
    )


def _install_probes() -> None:
    """Runs on every rank before the server starts; ``_probes`` returns what it records.

    ``steps``: per serving step (warmups and the KV-cache estimation are no steps),
    ``(context requests, other generation requests than attention-DP dummies, attention-DP
    dummies, batch size of the graph it replayed or None, whether the execution stream still ran
    earlier work when the step began)``.  ``guarded``: per graph replay, whether the dense replay
    guard (``DenseDwdpManager.register_replay_guard``) is among the runner's pre-replay hooks.
    ``dense``: the dense runtime's ``_COUNTERS`` and its launcher's transfer (``copy``) when the
    manager releases it.  The request queue waits for requests only in its first fetch.
    """
    from tensorrt_llm._torch.modules.dwdp.dense.manager import DenseDwdpManager
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner
    from tensorrt_llm._torch.pyexecutor.executor_request_queue import ExecutorRequestQueue
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    record = {"steps": [], "replays": [], "guarded": [], "dense": None}
    replay = CUDAGraphRunner.replay
    forward_step = PyExecutor._forward_step
    cleanup = DenseDwdpManager.cleanup
    fetch = ExecutorRequestQueue.get_from_request_queue

    def recorded_replay(self, key, inputs, **kwargs):
        record["replays"].append(key[0])
        guard = DenseDwdpManager._replay_hook
        hooks = self._pre_replay_hooks
        record["guarded"].append(any(getattr(hook, "__func__", None) is guard for hook in hooks))
        return replay(self, key, inputs, **kwargs)

    def recorded_forward_step(self, scheduled_requests, *args, **kwargs):
        replays = len(record["replays"])
        in_flight = not self.execution_stream.query()
        outputs = forward_step(self, scheduled_requests, *args, **kwargs)
        if not self.is_warmup:
            context = len(scheduled_requests.context_requests)
            requests = scheduled_requests.generation_requests
            dummies = sum(request.is_attention_dp_dummy for request in requests)
            graph = record["replays"][-1] if len(record["replays"]) > replays else None
            record["steps"].append((context, len(requests) - dummies, dummies, graph, in_flight))
        return outputs

    def recorded_cleanup(self):
        if self.dense is not None:
            record["dense"] = {name: getattr(self.dense, name) for name in _COUNTERS}
            record["dense"]["copy"] = self.dense._launcher.mode.value
        cleanup(self)

    def first_fetch_waits(self, timeout):
        # The batch wait applies to every fetch: it would hold every step back.
        items = fetch(self, timeout)
        if items:
            self.batch_wait_timeout_ms = 0
        return items

    CUDAGraphRunner.replay = recorded_replay
    PyExecutor._forward_step = recorded_forward_step
    DenseDwdpManager.cleanup = recorded_cleanup
    ExecutorRequestQueue.get_from_request_queue = first_fetch_waits
    PyExecutor.dense_e2e_record = record


def _probes() -> dict:
    """Runs on every rank after the server shut down; adds the CuTe DSL SM reserve
    (``sm_reserve``) and the gather's variables (``knobs``) to what the probes recorded."""
    import os

    from tensorrt_llm._torch.cute_dsl_utils import sm_reserve
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    record = PyExecutor.dense_e2e_record
    record["sm_reserve"] = sm_reserve()
    record["knobs"] = {name: os.environ.get(name) for name in _GATHER_KNOBS}
    return record


def _serve(target: Path, drafter: Path, env: dict[str, str]) -> _Run:
    """Serve the requests on fresh workers with ``env``."""
    session = MpiPoolSession(n_workers=_NUM_RANKS, env_overrides=env)
    try:
        submit_sync_per_worker(session, _install_probes)
        llm = LLM(
            str(target),
            tensor_parallel_size=_NUM_RANKS,
            moe_expert_parallel_size=_NUM_RANKS,
            enable_attention_dp=True,
            trust_remote_code=True,
            # More than the requests, so the first fetch from the request queue waits out the
            # timeout and takes them all.
            max_batch_size=16,
            batch_wait_timeout_ms=500,
            max_num_tokens=512,
            max_seq_len=1024,
            enable_autotuner=False,
            cuda_graph_config=CudaGraphConfig(batch_sizes=_GRAPH_BATCH_SIZES, enable_padding=True),
            moe_config=MoeConfig(backend="MEGAMOE_CUTEDSL", use_low_precision_moe_combine=True),
            kv_cache_config=KvCacheConfig(
                dtype="fp8",
                use_kv_cache_manager_v2=True,
                enable_block_reuse=False,
                free_gpu_memory_fraction=0.5,
                host_cache_size=0,
                tokens_per_block=64,
                mamba_ssm_cache_dtype="bfloat16",
            ),
            speculative_config=DSparkDecodingConfig(max_draft_len=5, speculative_model=drafter),
            _mpi_session=session,
        )
        with llm:
            pinned = [(rank, length) for rank, lengths in _REQUESTS.items() for length in lengths]
            results = llm.generate(
                list(_PROMPTS[: len(pinned)]),
                [SamplingParams(max_tokens=length, temperature=0) for _, length in pinned],
                scheduling_params=[
                    SchedulingParams(attention_dp_rank=rank, attention_dp_relax=False)
                    for rank, _ in pinned
                ],
            )
            kv_tokens = llm.get_kv_cache_capacity()["maxNumTokens"]
        ranks = submit_sync_per_worker(session, _probes)
    finally:
        session.shutdown()
    outputs = [(list(r.outputs[0].token_ids), tuple(r.spec_dec_totals)) for r in results]
    return _Run(outputs, kv_tokens, ranks)


@pytest.fixture(scope="module")
def checkpoints(device, tmp_path_factory) -> tuple[Path, Path]:
    """The cut target checkpoint and the remapped drafter."""
    if torch.cuda.device_count() < _NUM_RANKS:
        pytest.skip(f"needs {_NUM_RANKS} GPUs, found {torch.cuda.device_count()}")
    root = llm_models_root()
    target = root / _TARGET_CHECKPOINT if root else None
    drafters = (
        [root / name for name in _DRAFTER_CHECKPOINTS if (root / name).is_dir()] if root else []
    )
    if target is None or not target.is_dir() or not drafters:
        pytest.skip(
            f"needs {_TARGET_CHECKPOINT} and {' or '.join(_DRAFTER_CHECKPOINTS)} under "
            f"LLM_MODELS_ROOT ({root})"
        )
    return _checkpoints(target, drafters[0], tmp_path_factory.mktemp("checkpoints"))


@pytest.fixture(scope="module", params=["gather", "batched"])
def reference(request, device, checkpoints) -> tuple[str, Path, Path, _Run]:
    """The transfer, the checkpoints and the transfer's replicated run, which a second replicated
    run must reproduce."""
    transfer = request.param
    if transfer == "gather":
        _require_gather(device)
    target, drafter = checkpoints
    replicated = _serve(target, drafter, _env(transfer))
    assert _serve(target, drafter, _env(transfer)).outputs == replicated.outputs, (
        "the replicated runs differ, so their tokens are no reference for the dense runs"
    )
    return transfer, target, drafter, replicated


@pytest.mark.parametrize("debug", [2, 0], ids=["debug2", "debug0"])
def test_dense_dwdp_keeps_the_tokens_of_a_generation_server(reference, debug):
    """Dense DWDP: the replicated run's tokens and accepted drafts; see the module.

    At ``DEBUG=2`` the slot watchdog's poll after every step synchronizes the host; only at
    ``DEBUG=0`` does the overlap scheduler enqueue a step while the one before still runs.  With
    the copy engines no rank of any run has an SM reserve or a setting of the gather.
    """
    transfer, target, drafter, replicated = reference
    dense = _serve(target, drafter, _env(transfer, debug))

    # The dense run stepped eagerly with a batch above the largest graph, replayed padded
    # batches, and replayed attention-DP dummies on a rank without requests while the others
    # served theirs (the ranks step in lockstep, so step i of each rank is one iteration).
    steps = [step for rank in dense.ranks for step in rank["steps"]]
    largest = max(_GRAPH_BATCH_SIZES)
    assert any(not ctx and gen > largest and graph is None for ctx, gen, _, graph, _ in steps)
    assert any(graph and graph > gen + dummies for _, gen, dummies, graph, _ in steps)
    assert any(
        any(graph and dummies and not gen for _, gen, dummies, graph, _ in iteration)
        and any(gen for _, gen, *_ in iteration)
        for iteration in zip(*(rank["steps"] for rank in dense.ranks))
    )
    assert dense.outputs == replicated.outputs
    assert dense.kv_tokens > replicated.kv_tokens
    for rank in dense.ranks:
        assert any(graph for *_, graph, _ in rank["steps"])
        assert debug >= 2 or any(in_flight for *_, in_flight in rank["steps"])
        # Every replay ran behind the replay guard the model registers with its graph runner.
        assert rank["guarded"] and all(rank["guarded"])
        counters = rank["dense"]
        assert counters["copy"] == transfer
        assert counters["captures"] > 0 and (counters["watchdog_checks"] > 0) is (debug >= 2)
        assert counters["recoveries"] == counters["demand_fetches"] == 0
        assert counters["war_fences"] == counters["watchdog_mismatches"] == 0
    for rank in replicated.ranks:
        assert rank["guarded"] and not any(rank["guarded"])
    for run in (replicated, dense):
        for rank in run.ranks:
            if transfer == "batched":
                assert rank["sm_reserve"] == 0
                assert rank["knobs"] == dict.fromkeys(_GATHER_KNOBS)
            else:
                assert rank["sm_reserve"] > 0
                assert rank["knobs"] == _GATHER_ENV
