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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/moeGatherScatterKernels.h"

// Include NVFP4 support if available
#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif

namespace tensorrt_llm
{
namespace kernels
{

// 128-bit vector load/store configuration
constexpr int kVectorWidthBytes = 16; // 128 bits = 16 bytes
using VectorType = uint4;             // 16 bytes vector type

// Map sizeof(T) to appropriate unsigned integer type
template <int TypeSize>
struct UIntTypeMap;

template <>
struct UIntTypeMap<1>
{
    using type = uint8_t;
};

template <>
struct UIntTypeMap<2>
{
    using type = uint16_t;
};

template <>
struct UIntTypeMap<4>
{
    using type = uint32_t;
};

template <>
struct UIntTypeMap<8>
{
    using type = uint64_t;
};

// Helper to get the appropriate uint type for a given data type
template <typename T>
using UIntType = typename UIntTypeMap<sizeof(T)>::type;

// Calculate elements per vector for different type sizes
template <int TypeSize>
struct VectorConfig
{
    static constexpr int kElementsPerVector = kVectorWidthBytes / TypeSize;
};

// Note: VectorConfig<1> uses the default implementation:
// 16 bytes / 1 byte = 16 elements per vector
// This efficiently handles all 1-byte types including FP8 and NVFP4 (packed)

// Helper function to check if pointer is aligned for 128-bit access
__device__ __forceinline__ bool isAligned128(void const* ptr)
{
    return reinterpret_cast<uintptr_t>(ptr) % kVectorWidthBytes == 0;
}

// Generic gather kernel using uint types
template <int TypeSize>
__global__ void moeGatherKernelVectorized128Generic(
    void* output, void const* input, int const* indices, int numElements, int featureSize)
{
    using UIntT = typename UIntTypeMap<TypeSize>::type;
    constexpr int kElementsPerVector = VectorConfig<TypeSize>::kElementsPerVector;

    int const elementIdx = blockIdx.x;

    if (elementIdx >= numElements)
        return;

    int const srcIdx = indices[elementIdx];
    if (srcIdx < 0)
        return;

    UIntT const* srcPtr = static_cast<UIntT const*>(input) + srcIdx * featureSize;
    UIntT* dstPtr = static_cast<UIntT*>(output) + elementIdx * featureSize;

    int const tid = threadIdx.x;
    int const blockSize = blockDim.x;

    // Check if we can use 128-bit vectorized access
    bool const canUseVector = (featureSize % kElementsPerVector == 0) && isAligned128(srcPtr) && isAligned128(dstPtr);

    if (canUseVector)
    {
        // Use 128-bit vectorized loads/stores
        VectorType const* srcVec = reinterpret_cast<VectorType const*>(srcPtr);
        VectorType* dstVec = reinterpret_cast<VectorType*>(dstPtr);
        int const vectorCount = featureSize / kElementsPerVector;

        for (int i = tid; i < vectorCount; i += blockSize)
        {
            dstVec[i] = srcVec[i];
        }
    }
    else
    {
        // Fall back to scalar access
        for (int i = tid; i < featureSize; i += blockSize)
        {
            dstPtr[i] = srcPtr[i];
        }
    }
}

// Generic scatter kernel using uint types
template <int TypeSize>
__global__ void moeScatterKernelVectorized128Generic(
    void* output, void const* input, int const* indices, int numElements, int featureSize)
{
    using UIntT = typename UIntTypeMap<TypeSize>::type;
    constexpr int kElementsPerVector = VectorConfig<TypeSize>::kElementsPerVector;

    int const elementIdx = blockIdx.x;

    if (elementIdx >= numElements)
        return;

    int const dstIdx = indices[elementIdx];
    if (dstIdx < 0)
        return;

    UIntT const* srcPtr = static_cast<UIntT const*>(input) + elementIdx * featureSize;
    UIntT* dstPtr = static_cast<UIntT*>(output) + dstIdx * featureSize;

    int const tid = threadIdx.x;
    int const blockSize = blockDim.x;

    // Check if we can use 128-bit vectorized access
    bool const canUseVector = (featureSize % kElementsPerVector == 0) && isAligned128(srcPtr) && isAligned128(dstPtr);

    if (canUseVector)
    {
        // Use 128-bit vectorized loads/stores
        VectorType const* srcVec = reinterpret_cast<VectorType const*>(srcPtr);
        VectorType* dstVec = reinterpret_cast<VectorType*>(dstPtr);
        int const vectorCount = featureSize / kElementsPerVector;

        for (int i = tid; i < vectorCount; i += blockSize)
        {
            dstVec[i] = srcVec[i];
        }
    }
    else
    {
        // Fall back to scalar access
        for (int i = tid; i < featureSize; i += blockSize)
        {
            dstPtr[i] = srcPtr[i];
        }
    }
}

// Fallback scalar kernels
template <int TypeSize>
__global__ void moeGatherKernelScalarGeneric(
    void* output, void const* input, int const* indices, int numElements, int featureSize)
{
    using UIntT = typename UIntTypeMap<TypeSize>::type;

    int const idx = blockIdx.x * blockDim.x + threadIdx.x;
    int const elementIdx = idx / featureSize;
    int const featureIdx = idx % featureSize;

    if (elementIdx < numElements && featureIdx < featureSize)
    {
        int const srcIdx = indices[elementIdx];
        if (srcIdx >= 0)
        {
            UIntT const* srcPtr = static_cast<UIntT const*>(input);
            UIntT* dstPtr = static_cast<UIntT*>(output);

            dstPtr[elementIdx * featureSize + featureIdx] = srcPtr[srcIdx * featureSize + featureIdx];
        }
    }
}

template <int TypeSize>
__global__ void moeScatterKernelScalarGeneric(
    void* output, void const* input, int const* indices, int numElements, int featureSize)
{
    using UIntT = typename UIntTypeMap<TypeSize>::type;

    int const idx = blockIdx.x * blockDim.x + threadIdx.x;
    int const elementIdx = idx / featureSize;
    int const featureIdx = idx % featureSize;

    if (elementIdx < numElements && featureIdx < featureSize)
    {
        int const dstIdx = indices[elementIdx];
        if (dstIdx >= 0)
        {
            UIntT const* srcPtr = static_cast<UIntT const*>(input);
            UIntT* dstPtr = static_cast<UIntT*>(output);

            dstPtr[dstIdx * featureSize + featureIdx] = srcPtr[elementIdx * featureSize + featureIdx];
        }
    }
}

// Dispatch function to select the appropriate kernel based on type size
template <typename T>
void invokeMoeGather(
    T* output, T const* input, int const* indices, int numElements, int featureSize, cudaStream_t stream)
{
    if (numElements == 0 || featureSize == 0)
        return;

    constexpr int typeSize = sizeof(T);
    constexpr int kElementsPerVector = VectorConfig<typeSize>::kElementsPerVector;
    constexpr int kMinFeatureSizeForVectorization = kElementsPerVector * 2; // At least 2 vectors per element

    if (featureSize >= kMinFeatureSizeForVectorization)
    {
        // Use 128-bit vectorized kernel with one block per element
        constexpr int kThreadsPerBlock = 256;
        int const numBlocks = numElements;

        if constexpr (typeSize == 1)
        {
            moeGatherKernelVectorized128Generic<1>
                <<<numBlocks, kThreadsPerBlock, 0, stream>>>(output, input, indices, numElements, featureSize);
        }
        else if constexpr (typeSize == 2)
        {
            moeGatherKernelVectorized128Generic<2>
                <<<numBlocks, kThreadsPerBlock, 0, stream>>>(output, input, indices, numElements, featureSize);
        }
        else if constexpr (typeSize == 4)
        {
            moeGatherKernelVectorized128Generic<4>
                <<<numBlocks, kThreadsPerBlock, 0, stream>>>(output, input, indices, numElements, featureSize);
        }
        else if constexpr (typeSize == 8)
        {
            moeGatherKernelVectorized128Generic<8>
                <<<numBlocks, kThreadsPerBlock, 0, stream>>>(output, input, indices, numElements, featureSize);
        }
    }
    else
    {
        // Use scalar kernel for small feature sizes
        constexpr int kThreadsPerBlock = 256;
        int const totalThreads = numElements * featureSize;
        int const numBlocks = (totalThreads + kThreadsPerBlock - 1) / kThreadsPerBlock;

        if constexpr (typeSize == 1)
        {
            moeGatherKernelScalarGeneric<1>
                <<<numBlocks, kThreadsPerBlock, 0, stream>>>(output, input, indices, numElements, featureSize);
        }
        else if constexpr (typeSize == 2)
        {
            moeGatherKernelScalarGeneric<2>
                <<<numBlocks, kThreadsPerBlock, 0, stream>>>(output, input, indices, numElements, featureSize);
        }
        else if constexpr (typeSize == 4)
        {
            moeGatherKernelScalarGeneric<4>
                <<<numBlocks, kThreadsPerBlock, 0, stream>>>(output, input, indices, numElements, featureSize);
        }
        else if constexpr (typeSize == 8)
        {
            moeGatherKernelScalarGeneric<8>
                <<<numBlocks, kThreadsPerBlock, 0, stream>>>(output, input, indices, numElements, featureSize);
        }
    }
}

template <typename T>
void invokeMoeScatter(
    T* output, T const* input, int const* indices, int numElements, int featureSize, cudaStream_t stream)
{
    if (numElements == 0 || featureSize == 0)
        return;

    constexpr int typeSize = sizeof(T);
    constexpr int kElementsPerVector = VectorConfig<typeSize>::kElementsPerVector;
    constexpr int kMinFeatureSizeForVectorization = kElementsPerVector * 2; // At least 2 vectors per element

    if (featureSize >= kMinFeatureSizeForVectorization)
    {
        // Use 128-bit vectorized kernel with one block per element
        constexpr int kThreadsPerBlock = 256;
        int const numBlocks = numElements;

        if constexpr (typeSize == 1)
        {
            moeScatterKernelVectorized128Generic<1>
                <<<numBlocks, kThreadsPerBlock, 0, stream>>>(output, input, indices, numElements, featureSize);
        }
        else if constexpr (typeSize == 2)
        {
            moeScatterKernelVectorized128Generic<2>
                <<<numBlocks, kThreadsPerBlock, 0, stream>>>(output, input, indices, numElements, featureSize);
        }
        else if constexpr (typeSize == 4)
        {
            moeScatterKernelVectorized128Generic<4>
                <<<numBlocks, kThreadsPerBlock, 0, stream>>>(output, input, indices, numElements, featureSize);
        }
        else if constexpr (typeSize == 8)
        {
            moeScatterKernelVectorized128Generic<8>
                <<<numBlocks, kThreadsPerBlock, 0, stream>>>(output, input, indices, numElements, featureSize);
        }
    }
    else
    {
        // Use scalar kernel for small feature sizes
        constexpr int kThreadsPerBlock = 256;
        int const totalThreads = numElements * featureSize;
        int const numBlocks = (totalThreads + kThreadsPerBlock - 1) / kThreadsPerBlock;

        if constexpr (typeSize == 1)
        {
            moeScatterKernelScalarGeneric<1>
                <<<numBlocks, kThreadsPerBlock, 0, stream>>>(output, input, indices, numElements, featureSize);
        }
        else if constexpr (typeSize == 2)
        {
            moeScatterKernelScalarGeneric<2>
                <<<numBlocks, kThreadsPerBlock, 0, stream>>>(output, input, indices, numElements, featureSize);
        }
        else if constexpr (typeSize == 4)
        {
            moeScatterKernelScalarGeneric<4>
                <<<numBlocks, kThreadsPerBlock, 0, stream>>>(output, input, indices, numElements, featureSize);
        }
        else if constexpr (typeSize == 8)
        {
            moeScatterKernelScalarGeneric<8>
                <<<numBlocks, kThreadsPerBlock, 0, stream>>>(output, input, indices, numElements, featureSize);
        }
    }
}

// Explicit template instantiations for unsigned integer types
// These cover all data types through the generic UIntTypeMap system:
// uint8_t:  covers FP8, NVFP4 (packed), and other 1-byte types
// uint16_t: covers half, bfloat16, and other 2-byte types
// uint32_t: covers float and other 4-byte types

template void invokeMoeGather<uint8_t>(
    uint8_t* output, uint8_t const* input, int const* indices, int numElements, int featureSize, cudaStream_t stream);

template void invokeMoeGather<uint16_t>(
    uint16_t* output, uint16_t const* input, int const* indices, int numElements, int featureSize, cudaStream_t stream);

template void invokeMoeGather<uint32_t>(
    uint32_t* output, uint32_t const* input, int const* indices, int numElements, int featureSize, cudaStream_t stream);

template void invokeMoeScatter<uint8_t>(
    uint8_t* output, uint8_t const* input, int const* indices, int numElements, int featureSize, cudaStream_t stream);

template void invokeMoeScatter<uint16_t>(
    uint16_t* output, uint16_t const* input, int const* indices, int numElements, int featureSize, cudaStream_t stream);

template void invokeMoeScatter<uint32_t>(
    uint32_t* output, uint32_t const* input, int const* indices, int numElements, int featureSize, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
