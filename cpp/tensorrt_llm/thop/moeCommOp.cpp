/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/kernels/moeCommKernels.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <NvInferRuntime.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <vector>
#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#endif // ENABLE_MULTI_DEVICE

#include <cassert>
#include <set>

namespace torch_ext
{

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
moeCommPrepareIndicesOp(torch::Tensor gatheredTargetRankIds, c10::optional<torch::Tensor> realRankTokenCountCumSum,
    int64_t maxTokenCountPerRank, int64_t expertCount, int64_t topK, int64_t epRank, int64_t epSize)
{
    CHECK_INPUT(gatheredTargetRankIds, torch::kInt32);
    TORCH_CHECK(gatheredTargetRankIds.dim() == 2, "gatheredTargetRankIds must be a 2D tensor");
    TORCH_CHECK(gatheredTargetRankIds.size(1) == topK, "gatheredTargetRankIds must have topK columns");

    int const* realRankTokenCountCumSumPtr = nullptr;
    if (realRankTokenCountCumSum.has_value())
    {
        TORCH_CHECK(realRankTokenCountCumSum.value().dim() == 1, "realRankTokenCountCumSum must be a 1D tensor");
        TORCH_CHECK(realRankTokenCountCumSum.value().dtype() == torch::kInt32,
            "realRankTokenCountCumSum must be a int32 tensor");
        TORCH_CHECK(
            realRankTokenCountCumSum.value().size(0) == epSize, "realRankTokenCountCumSum must have epSize elements");
        realRankTokenCountCumSumPtr = realRankTokenCountCumSum.value().data_ptr<int>();
    }
    else
    {
        TORCH_CHECK(gatheredTargetRankIds.size(0) == epSize * maxTokenCountPerRank,
            "gatheredTargetRankIds should have shape (epSize * maxTokenCountPerRank, topK)");
    }
    TORCH_CHECK(maxTokenCountPerRank > 0, "maxTokenCountPerRank must be greater than 0");
    TORCH_CHECK(expertCount > 0, "expertCount must be greater than 0");
    TORCH_CHECK(topK > 0, "topK must be greater than 0");
    TORCH_CHECK(topK <= expertCount, "topK must be less than or equal to expertCount");
    TORCH_CHECK(epRank >= 0 && epRank < epSize, "epRank must be in the range [0, epSize)");

    auto stream = at::cuda::getCurrentCUDAStream();

    int maxSendRanksPerToken = std::max(epSize, topK);

    torch::Tensor localGatherIndices
        = torch::empty({maxTokenCountPerRank * epSize}, gatheredTargetRankIds.options().dtype(torch::kInt32));
    torch::Tensor sendRankCountCumSum = torch::empty({epSize}, gatheredTargetRankIds.options().dtype(torch::kInt32));
    torch::Tensor sendRankLocalIndices = torch::empty(
        {maxTokenCountPerRank * maxSendRanksPerToken}, gatheredTargetRankIds.options().dtype(torch::kInt32));
    torch::Tensor recvRankCountCumSum = torch::empty({epSize}, gatheredTargetRankIds.options().dtype(torch::kInt32));
    torch::Tensor recvRankLocalIndices
        = torch::empty({maxTokenCountPerRank * epSize}, gatheredTargetRankIds.options().dtype(torch::kInt32));
    torch::Tensor backwardRecvRankLocalIndices = torch::empty(
        {maxTokenCountPerRank * maxSendRanksPerToken}, gatheredTargetRankIds.options().dtype(torch::kInt32));

    tensorrt_llm::kernels::MoeExpertParallelInfo expertParallelInfo;
    expertParallelInfo.expertCount = expertCount;
    expertParallelInfo.topK = topK;

    tensorrt_llm::kernels::MoeEpWorldInfo worldInfo = {static_cast<int>(epSize), static_cast<int>(epRank)};
    tensorrt_llm::kernels::moeAllToAllPrepareIndices(worldInfo, expertParallelInfo, maxTokenCountPerRank,
        gatheredTargetRankIds.data_ptr<int>(), realRankTokenCountCumSumPtr, localGatherIndices.data_ptr<int>(),
        sendRankCountCumSum.data_ptr<int>(), sendRankLocalIndices.data_ptr<int>(), recvRankCountCumSum.data_ptr<int>(),
        recvRankLocalIndices.data_ptr<int>(), backwardRecvRankLocalIndices.data_ptr<int>(), stream);

    return std::make_tuple(localGatherIndices, sendRankCountCumSum, sendRankLocalIndices, recvRankCountCumSum,
        recvRankLocalIndices, backwardRecvRankLocalIndices);
}

void moeLocalGatherOp(torch::Tensor recvRankCumSum, torch::Tensor localGatherIndices, torch::Tensor gatheredExpertIds,
    torch::Tensor gatheredScales, torch::Tensor localExpertIds, torch::Tensor localScales, int64_t maxTokenCountPerRank,
    int64_t expertCount, int64_t topK, int64_t epRank, int64_t epSize)
{
    CHECK_INPUT(recvRankCumSum, torch::kInt32);
    CHECK_INPUT(localGatherIndices, torch::kInt32);
    CHECK_INPUT(gatheredExpertIds, torch::kInt32);
    CHECK_INPUT(gatheredScales, torch::kFloat32);
    CHECK_INPUT(localExpertIds, torch::kInt32);
    CHECK_INPUT(localScales, torch::kFloat32);

    TORCH_CHECK(maxTokenCountPerRank > 0, "maxTokenCountPerRank must be greater than 0");
    TORCH_CHECK(expertCount > 0, "expertCount must be greater than 0");
    TORCH_CHECK(topK > 0, "topK must be greater than 0");
    TORCH_CHECK(topK <= expertCount, "topK must be less than or equal to expertCount");
    TORCH_CHECK(epRank >= 0 && epRank < epSize, "epRank must be in the range [0, epSize)");

    TORCH_CHECK(recvRankCumSum.dim() == 1, "recvRankCumSum must be a 1D tensor");
    TORCH_CHECK(recvRankCumSum.size(0) == epSize, "recvRankCumSum must have epSize elements");
    TORCH_CHECK(localGatherIndices.dim() == 1, "localGatherIndices must be a 1D tensor");
    TORCH_CHECK(gatheredExpertIds.dim() == 2, "gatheredExpertIds must be a 2D tensor");
    TORCH_CHECK(gatheredScales.dim() == 2, "gatheredScales must be a 2D tensor");
    TORCH_CHECK(localExpertIds.dim() == 2, "localExpertIds must be a 2D tensor");
    TORCH_CHECK(localScales.dim() == 2, "localScales must be a 2D tensor");
    TORCH_CHECK(gatheredExpertIds.size(1) == topK, "gatheredExpertIds must have topK columns");
    TORCH_CHECK(gatheredScales.size(1) == topK, "gatheredScales must have topK columns");
    TORCH_CHECK(localExpertIds.size(1) == topK, "localExpertIds must have topK columns");
    TORCH_CHECK(localScales.size(1) == topK, "localScales must have topK columns");

    int localMaxTokenCount = static_cast<int>(localGatherIndices.size(0));
    TORCH_CHECK(localExpertIds.size(0) == localMaxTokenCount, "localExpertIds must have localMaxTokenCount rows");
    TORCH_CHECK(localScales.size(0) == localMaxTokenCount, "localScales must have localMaxTokenCount rows");

    auto stream = at::cuda::getCurrentCUDAStream();

    tensorrt_llm::kernels::MoeExpertParallelInfo expertParallelInfo;
    expertParallelInfo.expertCount = expertCount;
    expertParallelInfo.topK = topK;

    tensorrt_llm::kernels::MoeEpWorldInfo worldInfo = {static_cast<int>(epSize), static_cast<int>(epRank)};
    tensorrt_llm::kernels::moeLocalGather(worldInfo, expertParallelInfo, maxTokenCountPerRank, localMaxTokenCount,
        recvRankCumSum.data_ptr<int>(), localGatherIndices.data_ptr<int>(), gatheredExpertIds.data_ptr<int>(),
        gatheredScales.data_ptr<float>(), localExpertIds.data_ptr<int>(), localScales.data_ptr<float>(), stream);
}

void moeCommOp(torch::Tensor input, torch::Tensor sendRankCumSum, torch::Tensor sendIndices, torch::Tensor output,
    torch::Tensor recvRankCumSum, torch::Tensor recvIndices, torch::Tensor allWorkspaces, int64_t epRank,
    int64_t epSize, std::optional<bool> useNccl)
{
    CHECK_INPUT(sendRankCumSum, torch::kInt32);
    CHECK_INPUT(sendIndices, torch::kInt32);
    CHECK_INPUT(recvRankCumSum, torch::kInt32);
    CHECK_INPUT(recvIndices, torch::kInt32);
    // allWorkspaces is a uint64 tensor, but may not be contiguous
    TORCH_CHECK(allWorkspaces.dtype() == torch::kUInt64, "allWorkspaces must be a uint64 tensor");

    TORCH_CHECK(input.dim() == 2, "input must be a 2D tensor");
    TORCH_CHECK(output.dim() == 2, "output must be a 2D tensor");
    TORCH_CHECK(sendRankCumSum.dim() == 1, "sendRankCumSum must be a 1D tensor");
    TORCH_CHECK(sendIndices.dim() == 1, "sendIndices must be a 1D tensor");
    TORCH_CHECK(recvRankCumSum.dim() == 1, "recvRankCumSum must be a 1D tensor");
    TORCH_CHECK(recvIndices.dim() == 1, "recvIndices must be a 1D tensor");
    TORCH_CHECK(allWorkspaces.dim() == 2, "allWorkspaces must be a 2D tensor");

    TORCH_CHECK(input.size(1) == output.size(1), "input and output must have the same second dimension");
    TORCH_CHECK(sendRankCumSum.size(0) == epSize, "sendRankCumSum must have epSize elements");
    TORCH_CHECK(recvRankCumSum.size(0) == epSize, "recvRankCumSum must have epSize elements");
    TORCH_CHECK(allWorkspaces.size(0) == epSize, "allWorkspaces must have epSize elements");

    TORCH_CHECK(epRank >= 0 && epRank < epSize, "epRank must be in the range [0, epSize)");

    bool use_nccl = useNccl.has_value() ? useNccl.value() : false;

    if (use_nccl)
    {
#if ENABLE_MULTI_DEVICE
        TLLM_LOG_TRACE("%s start for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());

        torch::Tensor host_send_rank_cum_sum
            = torch::empty(sendRankCumSum.sizes(), sendRankCumSum.options().dtype(torch::kInt32));
        host_send_rank_cum_sum.copy_(sendRankCumSum);
        torch::Tensor host_send_indices = torch::empty(sendIndices.sizes(), sendIndices.options().dtype(torch::kInt32));
        host_send_indices.copy_(sendIndices);
        torch::Tensor host_recv_rank_cum_sum
            = torch::empty(recvRankCumSum.sizes(), recvRankCumSum.options().dtype(torch::kInt32));
        host_recv_rank_cum_sum.copy_(recvRankCumSum);
        torch::Tensor host_recv_indices = torch::empty(recvIndices.sizes(), recvIndices.options().dtype(torch::kInt32));
        host_recv_indices.copy_(recvIndices);

        // Synchronize to ensure host tensors are ready
        auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

        // Get NCCL communicator for EP group
        std::set<int> ep_group;
        for (int i = 0; i < epSize; i++)
        {
            ep_group.insert(i);
        }
        auto ncclComm = getComm(ep_group);
        TLLM_CHECK_WITH_INFO(ncclComm.get() != nullptr, "NCCL communicator should be initialized before use");

        // Get data type mapping
        auto type = tensorrt_llm::runtime::TorchUtils::dataType(input.scalar_type());
        auto ncclDataType = (*getDtypeMap())[type];

        // Get host data pointers
        int* host_send_counts = host_send_rank_cum_sum.data_ptr<int>();
        int* host_recv_counts = host_recv_rank_cum_sum.data_ptr<int>();

        // Convert cumulative sums to counts
        std::vector<int> send_counts(epSize);
        std::vector<int> recv_counts(epSize);
        std::vector<int> send_displs(epSize);
        std::vector<int> recv_displs(epSize);

        // Calculate send counts and displacements
        send_displs[0] = 0;
        send_counts[0] = host_send_counts[0];
        for (int i = 1; i < epSize; i++)
        {
            send_counts[i] = host_send_counts[i] - host_send_counts[i - 1];
            send_displs[i] = host_send_counts[i - 1];
        }

        // Calculate recv counts and displacements
        recv_displs[0] = 0;
        recv_counts[0] = host_recv_counts[0];
        for (int i = 1; i < epSize; i++)
        {
            recv_counts[i] = host_recv_counts[i] - host_recv_counts[i - 1];
            recv_displs[i] = host_recv_counts[i - 1];
        }

        // Get element size and total elements per vector
        size_t elt_size = input.dtype().itemsize();
        int64_t feature_size = input.size(1);

        // Create temporary buffers for gathering data according to indices
        torch::Tensor send_buffer = torch::empty({sendIndices.size(0), feature_size}, input.options());
        torch::Tensor recv_buffer = torch::empty({recvIndices.size(0), feature_size}, output.options());

        // Gather send data according to send indices
        int* send_indices_ptr = sendIndices.data_ptr<int>();
        for (int i = 0; i < sendIndices.size(0); i++)
        {
            if (send_indices_ptr[i] >= 0 && send_indices_ptr[i] < input.size(0))
            {
                send_buffer[i].copy_(input[send_indices_ptr[i]]);
            }
        }

        // Perform all-to-all using NCCL group operations
        ncclGroupStart();

        for (int peer = 0; peer < epSize; peer++)
        {
            if (peer != epRank)
            {
                // Send to peer
                if (send_counts[peer] > 0)
                {
                    void* send_ptr
                        = static_cast<char*>(send_buffer.data_ptr()) + send_displs[peer] * feature_size * elt_size;
                    NCCLCHECK_THROW(
                        ncclSend(send_ptr, send_counts[peer] * feature_size, ncclDataType, peer, *ncclComm, stream));
                }

                // Receive from peer
                if (recv_counts[peer] > 0)
                {
                    void* recv_ptr
                        = static_cast<char*>(recv_buffer.data_ptr()) + recv_displs[peer] * feature_size * elt_size;
                    NCCLCHECK_THROW(
                        ncclRecv(recv_ptr, recv_counts[peer] * feature_size, ncclDataType, peer, *ncclComm, stream));
                }
            }
            else
            {
                // Local copy for same rank
                if (send_counts[peer] > 0)
                {
                    void* send_ptr
                        = static_cast<char*>(send_buffer.data_ptr()) + send_displs[peer] * feature_size * elt_size;
                    void* recv_ptr
                        = static_cast<char*>(recv_buffer.data_ptr()) + recv_displs[peer] * feature_size * elt_size;
                    cudaMemcpyAsync(recv_ptr, send_ptr, send_counts[peer] * feature_size * elt_size,
                        cudaMemcpyDeviceToDevice, stream);
                }
            }
        }

        ncclGroupEnd();

        // Scatter received data according to recv indices
        int* recv_indices_ptr = recvIndices.data_ptr<int>();
        for (int i = 0; i < recvIndices.size(0); i++)
        {
            if (recv_indices_ptr[i] >= 0 && recv_indices_ptr[i] < output.size(0))
            {
                output[recv_indices_ptr[i]].copy_(recv_buffer[i]);
            }
        }

        TLLM_LOG_TRACE("%s stop for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
#else
        TORCH_CHECK(false, "NCCL support is not enabled. Please compile with ENABLE_MULTI_DEVICE=ON");
#endif // ENABLE_MULTI_DEVICE
        return;
    }

    tensorrt_llm::kernels::MoeEpWorldInfo worldInfo = {static_cast<int>(epSize), static_cast<int>(epRank)};
    tensorrt_llm::kernels::SendRecvDataInfo sendRecvDataInfo;

    size_t eltSize = input.dtype().itemsize();
    size_t eltCountPerU64 = sizeof(uint64_t) / eltSize;
    TORCH_CHECK(input.size(1) % (eltCountPerU64 * 2) == 0, "input.size(1) must be aligned to 16 bytes");
    sendRecvDataInfo.vectorSizeInU64 = input.size(1) / eltCountPerU64;
    sendRecvDataInfo.DoPreCompute();

    tensorrt_llm::kernels::SendRecvDispls sendDispls, recvDispls;
    sendDispls.dataPtr = static_cast<uint64_t*>(input.data_ptr());
    sendDispls.rankCountCumSum = sendRankCumSum.data_ptr<int>();
    sendDispls.rankLocalIndices = sendIndices.data_ptr<int>();
    sendDispls.vectorStrideInU64 = input.stride(0) / eltCountPerU64;

    recvDispls.dataPtr = static_cast<uint64_t*>(output.data_ptr());
    recvDispls.rankCountCumSum = recvRankCumSum.data_ptr<int>();
    recvDispls.rankLocalIndices = recvIndices.data_ptr<int>();
    recvDispls.vectorStrideInU64 = output.stride(0) / eltCountPerU64;

    tensorrt_llm::kernels::MoeCommWorkspace workspace;
    workspace.workspacePtr = allWorkspaces.data_ptr<uint64_t>();
    workspace.rankStrideInU64 = allWorkspaces.stride(0);

    auto stream = at::cuda::getCurrentCUDAStream();

    tensorrt_llm::kernels::moeAllToAll(worldInfo, sendRecvDataInfo, sendDispls, recvDispls, workspace, stream);
}

int64_t getWorkspaceSizePerRank(int64_t epSize)
{
    int epSize32 = static_cast<int>(epSize);
    return tensorrt_llm::kernels::getMoeCommWorkspaceSize(epSize32);
}

void setMaxUsableSmCount(int64_t maxSmCount)
{
    tensorrt_llm::kernels::setMaxUsableSmCount(maxSmCount);
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "moe_comm_prepare_indices(Tensor gathered_target_rank_ids, Tensor? real_rank_token_count_cum_sum, int "
        "max_token_count_per_rank, int expert_count, int top_k, int ep_rank, int ep_size) -> (Tensor, Tensor, Tensor, "
        "Tensor, "
        "Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("moe_comm_prepare_indices", &torch_ext::moeCommPrepareIndicesOp);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "moe_local_gather(Tensor recv_rank_cum_sum, Tensor local_gather_indices, Tensor gathered_expert_ids, Tensor "
        "gathered_scales, Tensor local_expert_ids, Tensor local_scales, int max_token_count_per_rank, int "
        "expert_count, int "
        "top_k, int ep_rank, int ep_size) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("moe_local_gather", &torch_ext::moeLocalGatherOp);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "moe_comm(Tensor input, Tensor send_rank_cum_sum, Tensor send_indices, Tensor output, Tensor "
        "recv_rank_cum_sum, "
        "Tensor recv_indices, Tensor all_workspaces, int ep_rank, int ep_size, bool? use_nccl=None) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("moe_comm", &torch_ext::moeCommOp);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("get_moe_commworkspace_size_per_rank(int ep_size) -> int");
}

TORCH_LIBRARY_IMPL(trtllm, CompositeExplicitAutograd, m)
{
    m.impl("get_moe_commworkspace_size_per_rank", &torch_ext::getWorkspaceSizePerRank);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("set_moe_max_usable_sm_count(int max_sm_count) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CompositeExplicitAutograd, m)
{
    m.impl("set_moe_max_usable_sm_count", &torch_ext::setMaxUsableSmCount);
}
