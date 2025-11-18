import pytest
import torch

from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3Gate


@pytest.mark.parametrize("seq_len", [1, 32, 128, 8192])
@pytest.mark.parametrize("num_experts, n_group, topk_group, top_k", [
    (256, 8, 4, 8),
    (72, 1, 1, 6),
    (384, 1, 1, 8),
])
@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.bfloat16, torch.float32])
def test_noaux_tc_run(seq_len, num_experts, n_group, topk_group, top_k, dtype):
    ROUTED_SCALING_FACTOR = 2.5
    HIDDEN_SIZE = 7168
    torch.manual_seed(24)
    torch.cuda.manual_seed(24)

    weight = torch.randn((num_experts, HIDDEN_SIZE), dtype=dtype).cuda()
    e_score_correction_bias = torch.randn((num_experts),
                                          dtype=torch.float32).cuda()

    logits = torch.randn((seq_len, HIDDEN_SIZE), dtype=dtype).cuda()

    weights = {}
    weights["weight"] = weight
    weights["e_score_correction_bias"] = e_score_correction_bias

    # Run the thop
    gate = DeepseekV3Gate(
        hidden_size=HIDDEN_SIZE,
        num_experts=num_experts,
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
        dtype=dtype,
        fuse_routing_kernel=True,
        apply_routing=False,
    )
    gate.load_weights([weights])
    gate.cuda()
    with torch.inference_mode():
        selected_indices, selected_values = gate.routing_method.apply(
            gate.forward(logits))

    # compare
    torch.cuda.synchronize()
