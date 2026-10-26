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
"""One rank of ``test_megamoe_genphase_tail_reset.py``: the tail-reset race, forced.

``python _megamoe_genphase_race_worker.py RANK PORT OUT_DIR``; the test starts one process per GPU.

Each rank builds the same EP4 MoE on ``MEGAMOE_CUTEDSL`` with the generation-phase kernel, behind a
test-only communication component (``_HoldingTokenComm``) whose holds are compile-time constants of
this rank's kernels: the zero-route rank holds its sort CTAs, and with them its ``metadata_ready``
and the tail of its main kernel, and the late-publisher rank holds the payload flags its pushers
send every rank. The production component is untouched; this process rebinds the name the main
kernel builds its component from.

The rank writes ``rank<R>.warm.json`` after a warm-up forward in which every rank receives routes
(it compiles the kernels and fences the workspaces, and no rank is without work, so it cannot
race), then runs the race forwards, in which no route reaches the zero-route rank, and one more
forward routed like the warm-up, which shows any counter the race left behind. It writes
``rank<R>.done.json`` once all of them finished and matched the reference. A hang shows as a
missing done file; the test owns the timeout.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import ClassVar

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int64, T, dsl_user_op
from cutlass.utils import HardwareInfo
from transformers.configuration_utils import PretrainedConfig

# ``tests/unittest``, for the shared MoE test utilities.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _torch.moe.quantize_utils import NVFP4QuantizeUtil, NVFP4RefMegaMoECuteDsl  # noqa: E402

from tensorrt_llm._torch.model_config import ModelConfig  # noqa: E402
from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import (  # noqa: E402
    megamoe_occupancy_sm_reserve,
)
from tensorrt_llm._torch.moe.fused_moe import RenormalizeMoeRoutingMethod, create_moe  # noqa: E402
from tensorrt_llm._torch.moe.fused_moe.mega_moe import MegaMoECuteDsl  # noqa: E402
from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.helpers.ptx_helpers import (  # noqa: E402
    nanosleep,
)
from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.kernel_src.rubin.inference.mega import (  # noqa: E402
    block_scaled_swap_ab_mega_moe_kernel_gen_specialized as gen_kernel_module,
)
from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.kernel_src.rubin.inference.mega.token_comm_gen_specialized import (  # noqa: E402, E501
    GenphaseTokenComm,
)
from tensorrt_llm.mapping import Mapping  # noqa: E402
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig  # noqa: E402

NUM_RANKS = 4
#: No route reaches this rank in the race forwards, so its main kernel owns no tile.
ZERO_ROUTE_RANK = 3
#: This rank's pushers publish their payload flags late, after the zero-route rank's tail.
LATE_PUBLISHER_RANK = 0
#: Holds the zero-route rank's ``metadata_ready``, so its tail lands inside the publication window:
#: after the other ranks' flags, before the late publisher's.
METADATA_HOLD_NS = 300_000
PUBLISH_HOLD_NS = 1_500_000
#: Large enough that the main kernel's grid, shrunk by it, is resident beside the communication grid,
#: so no cluster waits for a refine CTA to exit (which would order the tail after the publications).
SM_RESERVE = 46

NUM_EXPERTS = 32
TOP_K = 4
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 512
#: Tokens per rank, which is also the MegaMoE bucket (``max_num_tokens``).
NUM_TOKENS = 32
RACE_FORWARDS = 4
#: A router logit no expert of the zero-route rank can win a top-k slot with.
EXCLUDED_LOGIT = -1.0e4


@dsl_user_op
def _globaltimer_ns(*, loc=None, ip=None) -> Int64:
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "mov.u64 $0, %globaltimer;",
            "=l",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _hold(nanoseconds: int) -> None:
    """Keep the calling thread here for at least ``nanoseconds`` of GPU global time."""
    start = _globaltimer_ns()
    now = _globaltimer_ns()
    while now - start < Int64(nanoseconds):
        nanosleep(1000)
        now = _globaltimer_ns()


class _HoldingTokenComm(GenphaseTokenComm):
    """``GenphaseTokenComm`` that holds back one of this rank's publications (test-only).

    ``metadata_hold_ns`` holds the sort CTAs before they sort, so this rank's ``metadata_ready``;
    ``publish_hold_ns`` holds the pushers between pushing and publishing their payload flags. Every
    other step, and every other rank's component, is the production one.
    """

    metadata_hold_ns: ClassVar[int] = 0
    publish_hold_ns: ClassVar[int] = 0
    #: Components built in this process, so the run can show the main kernel built this one.
    built: ClassVar[int] = 0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        type(self).built += 1

    @cute.jit
    def _run_sort(self, device_workspace, local_rank, smem_base, source_rank, thread_idx) -> None:
        if cutlass.const_expr(self.metadata_hold_ns > 0):
            _hold(self.metadata_hold_ns)
        GenphaseTokenComm._run_sort(
            self, device_workspace, local_rank, smem_base, source_rank, thread_idx
        )

    @cute.jit
    def _publish_payload_flags(self, device_workspace, peer_rank_ptr_mapper, thread_idx) -> None:
        if cutlass.const_expr(self.publish_hold_ns > 0):
            _hold(self.publish_hold_ns)
        GenphaseTokenComm._publish_payload_flags(
            self, device_workspace, peer_rank_ptr_mapper, thread_idx
        )


def _check_preconditions(backend) -> None:
    """Fail unless the warm-up ran what the race needs; otherwise the test could pass vacuously."""
    if NUM_TOKENS not in backend._genphase_direct_input_buckets or _HoldingTokenComm.built == 0:
        raise RuntimeError("the generation-phase kernel with the test's component did not run")
    if megamoe_occupancy_sm_reserve() != SM_RESERVE:
        raise RuntimeError(f"MegaMoE sized its grid with reserve {megamoe_occupancy_sm_reserve()}")
    # Necessary, not sufficient: the clusters also have to fit around the communication grid.
    kernel = gen_kernel_module.BlockScaledSwapAbGenphaseMoeKernel
    cluster_size = kernel.cluster_shape_mn[0] * kernel.cluster_shape_mn[1]
    main_ctas = cluster_size * HardwareInfo().get_max_active_clusters(cluster_size)
    comm_ctas = NUM_RANKS + 1 + kernel.pusher_cta_count
    sm_count = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    if main_ctas + comm_ctas > sm_count:
        raise RuntimeError(
            f"{main_ctas} main and {comm_ctas} communication CTAs exceed {sm_count} SMs: the main "
            f"grid cannot sit beside the communication grid, raise SM_RESERVE"
        )


def _write(out_dir: Path, name: str, payload: dict) -> None:
    tmp = out_dir / f".{name}.tmp"
    tmp.write_text(json.dumps(payload))
    tmp.rename(out_dir / name)


def _build(rank: int):
    mapping = Mapping(
        world_size=NUM_RANKS,
        rank=rank,
        tp_size=NUM_RANKS,
        moe_ep_size=NUM_RANKS,
        moe_tp_size=1,
        enable_attention_dp=True,
    )
    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = NUM_EXPERTS
    pretrained_config.hidden_size = HIDDEN_SIZE
    pretrained_config.intermediate_size = INTERMEDIATE_SIZE
    pretrained_config.torch_dtype = torch.bfloat16
    quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        mapping=mapping,
        quant_config=quant_config,
        moe_backend="MEGAMOE_CUTEDSL",
        max_num_tokens=NUM_TOKENS,
    )
    routing_method = RenormalizeMoeRoutingMethod(top_k=TOP_K)
    # Same seed on every rank: every rank loads its slice of the same experts.
    torch.manual_seed(0)
    x = torch.randn((NUM_TOKENS, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda")
    quantize_util = NVFP4QuantizeUtil(
        num_experts=NUM_EXPERTS,
        dtype=torch.bfloat16,
        intermediate_size=INTERMEDIATE_SIZE,
        hidden_size=HIDDEN_SIZE,
        quant_config=quant_config,
        num_local_experts=NUM_EXPERTS // NUM_RANKS,
    )
    weights = quantize_util.create_weights(x_sf_global=(448 * 6) / x.abs().max().float())
    moe = create_moe(routing_method=routing_method, reduce_results=True, model_config=model_config)
    moe.load_weights([weights])
    moe.post_load_weights()
    moe.cuda(f"cuda:{rank}")
    reference = quantize_util.create_ref_module(routing_method, ref_cls=NVFP4RefMegaMoECuteDsl)
    reference.load_weights([weights])
    reference.cuda(f"cuda:{rank}")
    return moe, reference, routing_method, x


def main(rank: int, port: int, out_dir: Path) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(NUM_RANKS),
        LOCAL_RANK=str(rank),
        TRTLLM_CUTEDSL_SM_RESERVE=str(SM_RESERVE),
    )
    torch.cuda.set_device(rank)
    # Before any MegaMoE kernel exists: the main kernel resolves this name when it is built.
    _HoldingTokenComm.metadata_hold_ns = METADATA_HOLD_NS if rank == ZERO_ROUTE_RANK else 0
    _HoldingTokenComm.publish_hold_ns = PUBLISH_HOLD_NS if rank == LATE_PUBLISHER_RANK else 0
    gen_kernel_module.GenphaseTokenComm = _HoldingTokenComm
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=NUM_RANKS,
        device_id=torch.device("cuda", rank),
    )
    try:
        with torch.inference_mode():
            run(rank, out_dir)
    finally:
        dist.destroy_process_group()


def run(rank: int, out_dir: Path) -> None:
    moe, reference, routing_method, x = _build(rank)
    backend = moe.backend
    if not isinstance(backend, MegaMoECuteDsl):
        raise RuntimeError(f"expected the MegaMoECuteDsl backend, got {type(backend).__name__}")
    all_rank_num_tokens = [NUM_TOKENS] * NUM_RANKS
    generator = torch.Generator(device="cuda").manual_seed(1234 + rank)
    logits = torch.randn(
        (NUM_TOKENS, NUM_EXPERTS), dtype=torch.bfloat16, device="cuda", generator=generator
    )

    t0 = time.monotonic()
    warm = moe.forward(x, logits, all_rank_num_tokens=all_rank_num_tokens)
    torch.cuda.synchronize()
    routed_reference = reference.forward(x, logits)
    reference.check_accuracy(warm, routed_reference)
    _check_preconditions(backend)
    dist.barrier()
    _write(out_dir, f"rank{rank}.warm.json", {"setup_s": round(time.monotonic() - t0, 1)})

    experts_per_rank = NUM_EXPERTS // NUM_RANKS
    masked = logits.clone()
    zero_route_experts = slice(
        ZERO_ROUTE_RANK * experts_per_rank, (ZERO_ROUTE_RANK + 1) * experts_per_rank
    )
    masked[:, zero_route_experts] = EXCLUDED_LOGIT
    routed, _ = routing_method.apply(masked)
    if bool(((routed // experts_per_rank) == ZERO_ROUTE_RANK).any()):
        raise RuntimeError("a route still reaches the zero-route rank")
    # Line the ranks up, so that the holds alone decide when each publication lands.
    dist.barrier()

    t1 = time.monotonic()
    # The backend returns a view of its persistent combine buffer, which the next forward
    # overwrites, so each output is copied out before the next forward runs.
    outputs = [
        moe.forward(x, masked, all_rank_num_tokens=all_rank_num_tokens).clone()
        for _ in range(RACE_FORWARDS)
    ]
    torch.cuda.synchronize()
    race_s = time.monotonic() - t1
    # Every forward waits for the late publisher's flags on every rank, so the hold must show.
    if race_s < RACE_FORWARDS * PUBLISH_HOLD_NS / 1e9:
        raise RuntimeError(f"the forwards took {race_s:.4f} s: the publication hold did not apply")
    reference.check_accuracy(outputs[0], reference.forward(x, masked))
    # The separate top-k reduce is deterministic, so every forward must agree bit for bit.
    for output in outputs[1:]:
        if not torch.equal(output, outputs[0]):
            raise RuntimeError("race forwards disagree")
    # A counter the zero-route tails left behind bites only once that rank owns tiles again: a
    # refine claim landing after the reset, for one, makes the next launch's refine skip blocks
    # whose scale factors FC1 then waits for forever. So the last forward routes to every rank.
    rerouted = moe.forward(x, logits, all_rank_num_tokens=all_rank_num_tokens)
    torch.cuda.synchronize()
    reference.check_accuracy(rerouted, routed_reference)
    dist.barrier()
    _write(
        out_dir,
        f"rank{rank}.done.json",
        {
            "race_s": round(race_s, 4),
            "zero_route_rank": ZERO_ROUTE_RANK,
            "late_publisher_rank": LATE_PUBLISHER_RANK,
            "output_sum": float(outputs[0].float().sum()),
        },
    )


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]))
