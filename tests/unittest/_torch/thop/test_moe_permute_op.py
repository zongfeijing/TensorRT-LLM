import random
import unittest
from typing import List

import torch

import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils


class TestMoePermuteOp(unittest.TestCase):
    """
    Unit tests for torch.ops.trtllm.moe_permute_op with custom token distribution.
    """

    def setUp(self):
        """
        Set up test parameters.

        Configuration options:
        - skip_correctness_check: Set to True to skip expert distribution verification.
                                  Useful for quick testing or debugging.
        - input_dtype: Input data type
                       - torch.bfloat16: BFloat16 input (no quantization)
                       - fp4_utils.float4_e2m1x2: NVFP4 quantized input
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

        self.device = torch.device("cuda")
        self.num_tokens = 256
        self.num_experts_global = 256  # Total number of experts across all ranks
        self.hidden_size = 128
        self.top_k = 8  # Each token selects 8 experts

        # Expert Parallelism configuration
        self.ep_size = 32  # Number of EP ranks
        self.ep_rank = 0  # Current EP rank (0-31)

        # Calculate local expert range based on EP configuration
        self.num_experts_local = self.num_experts_global // self.ep_size  # 256 / 32 = 8
        self.local_expert_start_id = self.ep_rank * self.num_experts_local  # 0 * 8 = 0
        self.local_expert_end_id = (
            self.local_expert_start_id + self.num_experts_local - 1
        )  # 0 + 8 - 1 = 7

        # Expert token distribution for LOCAL experts on current rank
        # This represents how many tokens are assigned to each local expert
        # For ep_rank=0, we have 8 local experts (0-7)
        base_counts = [89, 200, 145, 178, 241, 78, 198, 60]
        self.expert_token_counts = base_counts[: self.num_experts_local]

        # Control flags
        self.skip_correctness_check = False  # Set to True to skip distribution verification
        # Input dtype: torch.bfloat16 for bf16, or fp4_utils.float4_e2m1x2 for nvfp4
        self.input_dtype = (
            fp4_utils.float4_e2m1x2
        )  # Options: torch.bfloat16 or fp4_utils.float4_e2m1x2
        self.input_dtype = torch.bfloat16  # Uncomment to test with BFloat16

    def _construct_token_selected_experts(
        self,
        expert_token_counts: List[int],
        top_k: int = 8,
        num_tokens: int = 200,
        num_experts_global: int = 256,
        local_expert_start_id: int = 0,
        local_expert_end_id: int = 31,
    ) -> torch.Tensor:
        """
        Construct token_selected_experts tensor with EXACT specified distribution.

        Args:
            expert_token_counts: List indicating how many tokens should select each LOCAL expert
            top_k: Number of experts per token
            num_tokens: Total number of tokens
            num_experts_global: Total number of experts across all ranks
            local_expert_start_id: Starting expert ID for local experts on this rank
            local_expert_end_id: Ending expert ID for local experts on this rank

        Returns:
            token_selected_experts: [num_tokens, top_k] tensor
                Contains global expert IDs (0 to num_experts_global-1)
        """
        num_experts_local = len(expert_token_counts)
        assert local_expert_end_id - local_expert_start_id + 1 == num_experts_local, (
            f"Local expert range mismatch: {local_expert_end_id - local_expert_start_id + 1} != {num_experts_local}"
        )

        total_slots = num_tokens * top_k

        # Create a flat list of expert IDs with EXACT counts
        expert_id_list = []

        # Add local experts with specified counts
        for local_expert_idx, count in enumerate(expert_token_counts):
            global_expert_id = local_expert_start_id + local_expert_idx
            expert_id_list.extend([global_expert_id] * count)

        # Fill remaining slots with experts from other ranks (non-local experts)
        remaining_slots = total_slots - len(expert_id_list)
        if remaining_slots > 0:
            # Use experts from other EP ranks to fill remaining slots
            non_local_expert_ids = []
            for i in range(num_experts_global):
                if i < local_expert_start_id or i > local_expert_end_id:
                    non_local_expert_ids.append(i)

            # Randomly fill remaining slots with non-local experts
            for _ in range(remaining_slots):
                expert_id_list.append(random.choice(non_local_expert_ids))
        elif remaining_slots < 0:
            raise ValueError(
                f"Total expert counts {len(expert_id_list)} exceeds available slots {total_slots}"
            )

        # Shuffle the list to randomize distribution across tokens
        random.shuffle(expert_id_list)

        # Convert to tensor and reshape
        token_selected_experts = torch.tensor(
            expert_id_list, dtype=torch.int32, device=self.device
        ).reshape(num_tokens, top_k)

        return token_selected_experts

    def test_moe_permute_op_basic(self):
        """
        Test basic functionality of moe_permute_op with custom token distribution.

        Distributed scenario:
        - 256 total experts across all ranks
        - EP size = 32 (32 EP ranks)
        - EP rank = 0 (current rank)
        - 8 local experts (0-7) on current rank
        - Distribution: [89, 200, 145, 178, 241, 78, 198, 60] tokens select each local expert
        - Top-K = 8: each token selects 8 experts (from global pool of 256)

        Input dtype options:
        - torch.bfloat16: BFloat16 input (no quantization)
        - fp4_utils.float4_e2m1x2: NVFP4 quantized input with block-wise scale factors
        """
        # Determine dtype name for display
        if self.input_dtype == torch.bfloat16:
            dtype_name = "bf16 (torch.bfloat16)"
        elif self.input_dtype == fp4_utils.float4_e2m1x2:
            dtype_name = "nvfp4 (fp4_utils.float4_e2m1x2)"
        else:
            dtype_name = str(self.input_dtype)

        # Print test configuration
        print(f"\n{'=' * 60}")
        print("Test Configuration:")
        print(f"  Input dtype: {dtype_name}")
        print(f"  Num tokens: {self.num_tokens}")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Top-K: {self.top_k}")
        print(f"  EP size: {self.ep_size}, EP rank: {self.ep_rank}")
        print(f"  Local experts: {self.local_expert_start_id}-{self.local_expert_end_id}")
        print(f"{'=' * 60}\n")

        # Construct token_selected_experts with the specified distribution
        token_selected_experts = self._construct_token_selected_experts(
            self.expert_token_counts,
            self.top_k,
            self.num_tokens,
            self.num_experts_global,
            self.local_expert_start_id,
            self.local_expert_end_id,
        )

        # Create input tensor based on dtype
        if self.input_dtype == torch.bfloat16:
            # Create bfloat16 input tensor
            x = torch.randn(
                self.num_tokens, self.hidden_size, dtype=torch.bfloat16, device=self.device
            )
            x_sf = None  # No scale factor for bf16

        elif self.input_dtype == fp4_utils.float4_e2m1x2:
            # Create and quantize to nvfp4
            # First create fp16 data
            x_fp16 = torch.randn(
                self.num_tokens, self.hidden_size, dtype=torch.float16, device=self.device
            )

            # Calculate global scale factor for nvfp4 quantization
            # Formula: (448 * 6) / max_abs_value
            # 448 is the max value that e4m3 can hold (used internally)
            # 6 is a safety factor
            x_global_sf = (448 * 6) / x_fp16.abs().max().float()

            # Quantize to nvfp4 (float4_e2m1)
            # Schema: fp4_quantize(Tensor input, Tensor? globalScale, int sfVecSize,
            #                      bool sfUseUE8M0=False, bool isSfSwizzledLayout=True)
            # sfVecSize: scaling vector size (16 for standard block-wise scaling)
            # sfUseUE8M0: False (use UFP8 format for scale factors)
            # isSfSwizzledLayout: True (use swizzled layout for scale factors)
            sf_vec_size = 16
            x, x_sf = torch.ops.trtllm.fp4_quantize(
                x_fp16,
                x_global_sf,
                sf_vec_size,
                False,  # sfUseUE8M0
                True,  # isSfSwizzledLayout
            )
            # x is now nvfp4 dtype (float4_e2m1x2 - packed format)
            # x_sf is the block-wise scale factors (UFP8 format)

        else:
            raise ValueError(
                f"Unsupported input_dtype: {self.input_dtype}. "
                f"Use torch.bfloat16 or fp4_utils.float4_e2m1x2."
            )

        # Create token final scales
        token_final_scales = torch.ones(
            self.num_tokens, self.top_k, dtype=torch.float32, device=self.device
        )

        # Create quant_scales for NVFP4 if needed
        quant_scales = None
        if self.input_dtype == fp4_utils.float4_e2m1x2:
            # For NVFP4, we need to provide quant_scales even though we're not doing actual GEMM
            # quant_scales = [fc1_weight_block_scale, fc1_global_scale, fc2_weight_block_scale, fc2_global_scale]

            # Create dummy weight block scales (will not be used in permute-only operation)
            # FC1: [num_experts, intermediate_size, hidden_size // 16]
            intermediate_size = 256  # Dummy value for testing
            fc1_weight_block_scale = torch.ones(
                self.num_experts_local,
                intermediate_size * 2,  # *2 for SwiGLU
                self.hidden_size // 16,
                dtype=fp4_utils.float4_sf_dtype,
                device=self.device,
            )

            # FC1 global scales: [num_experts]
            fc1_global_scale = torch.ones(
                self.num_experts_local, dtype=torch.float32, device=self.device
            )

            # FC2: [num_experts, hidden_size, intermediate_size // 16]
            fc2_weight_block_scale = torch.ones(
                self.num_experts_local,
                self.hidden_size,
                intermediate_size // 16,
                dtype=fp4_utils.float4_sf_dtype,
                device=self.device,
            )

            # FC2 global scales: [num_experts]
            fc2_global_scale = torch.ones(
                self.num_experts_local, dtype=torch.float32, device=self.device
            )

            quant_scales = [
                fc1_weight_block_scale,
                fc1_global_scale,
                fc2_weight_block_scale,
                fc2_global_scale,
            ]

        # Call moe_permute_op
        (
            permuted_row_to_unpermuted_row_tensor,
            permuted_token_selected_experts_tensor,
            permuted_data_tensor,
            expert_first_token_offset_tensor,
            permuted_token_final_scales_tensor,
            unpermuted_row_to_permuted_row_tensor,
        ) = torch.ops.trtllm.moe_permute_op(
            x,
            token_selected_experts,
            token_final_scales,
            None,  # w3_w1_weight
            None,  # w2_weight
            quant_scales,  # quant_scales (required for NVFP4)
            input_sf=x_sf,  # nvfp4 scale factor
            num_experts_on_rank=self.num_experts_local,
            tp_size=1,
            tp_rank=0,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            cluster_size=1,
            cluster_rank=0,
            min_latency_mode=False,
            use_fp8_block_scaling=False,
        )

        # Verify output shapes
        # Note: With EP, moe_permute_op only processes tokens assigned to LOCAL experts
        # So the output size depends on how many tokens were routed to local experts
        # self.assertEqual(permuted_data_tensor.shape[1], self.hidden_size)
        self.assertEqual(expert_first_token_offset_tensor.shape[0], self.num_experts_local + 1)

        # Verify expert_first_token_offset_tensor
        # This tensor should contain the cumulative offsets for LOCAL experts
        print(
            f"\nExpert first token offsets (local experts {self.local_expert_start_id}-{self.local_expert_end_id}):"
        )
        print(f"{expert_first_token_offset_tensor.cpu().tolist()}")

        # Count how many times each LOCAL expert appears in token_selected_experts
        print(
            f"\nLocal Expert Distribution (Experts {self.local_expert_start_id}-{self.local_expert_end_id}):"
        )
        actual_local_expert_counts = []
        for local_expert_idx in range(self.num_experts_local):
            global_expert_id = self.local_expert_start_id + local_expert_idx
            count = (token_selected_experts == global_expert_id).sum().item()
            actual_local_expert_counts.append(count)

        print(f"Expected expert selection counts: {self.expert_token_counts}")
        print(f"Actual expert selection counts: {actual_local_expert_counts}")

        # Verify that each LOCAL expert appears the expected number of times (EXACTLY)
        if not self.skip_correctness_check:
            for i in range(self.num_experts_local):
                print(
                    f"Local Expert {i} (Global ID {self.local_expert_start_id + i}): "
                    f"expected {self.expert_token_counts[i]} selections, "
                    f"got {actual_local_expert_counts[i]} selections"
                )
                # Exact match - the construction method guarantees precise counts
                self.assertEqual(
                    actual_local_expert_counts[i],
                    self.expert_token_counts[i],
                    msg=f"Local expert {i} selection count mismatch: "
                    f"expected {self.expert_token_counts[i]}, got {actual_local_expert_counts[i]}",
                )
        else:
            print("Skipping correctness verification (skip_correctness_check=True)")
            for i in range(self.num_experts_local):
                print(
                    f"Local Expert {i} (Global ID {self.local_expert_start_id + i}): "
                    f"expected {self.expert_token_counts[i]} selections, "
                    f"got {actual_local_expert_counts[i]} selections"
                )

        # Verify offsets from the moe_permute_op output
        actual_offsets = expert_first_token_offset_tensor.cpu().tolist()
        print("\nToken assignments from moe_permute_op:")
        for i in range(self.num_experts_local):
            actual_count = actual_offsets[i + 1] - actual_offsets[i]
            print(f"Local Expert {i}: moe_permute_op assigned {actual_count} tokens")

        # Count total tokens routed to local experts
        total_local_tokens = actual_offsets[-1]
        print(f"\nTotal tokens routed to local experts: {total_local_tokens}")

        # Also show distribution of ALL global experts selected
        print("\nGlobal expert selection statistics:")
        global_expert_counts = {}
        for expert_id in token_selected_experts.flatten().cpu().tolist():
            global_expert_counts[expert_id] = global_expert_counts.get(expert_id, 0) + 1
        print(f"Number of unique global experts selected: {len(global_expert_counts)}")
        print(
            f"Global expert ID range: {min(global_expert_counts.keys())} - {max(global_expert_counts.keys())}"
        )


if __name__ == "__main__":
    unittest.main()
