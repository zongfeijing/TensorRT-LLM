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

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif

namespace tensorrt_llm
{
namespace kernels
{

/*!
 * \brief Gather elements from input tensor according to indices using optimized CUDA kernels
 *
 * This function performs a gather operation where elements from the input tensor are collected
 * according to the specified indices and stored in the output tensor. The operation is optimized
 * with 128-bit vectorized memory operations when possible.
 *
 * The operation is equivalent to:
 * for (int i = 0; i < numElements; i++) {
 *     int srcIdx = indices[i];
 *     if (srcIdx >= 0) {
 *         output[i] = input[srcIdx];  // Copy entire feature vector
 *     }
 * }
 *
 * Supports various data types including:
 * - float (32-bit): 4 elements per 128-bit vector
 * - half (16-bit): 8 elements per 128-bit vector
 * - bfloat16 (16-bit): 8 elements per 128-bit vector
 * - FP8 types (8-bit): 16 elements per 128-bit vector
 * - NVFP4 types (4-bit, packed): 32 elements per 128-bit vector
 *
 * \param output Pointer to output tensor [numElements, featureSize]
 * \param input Pointer to input tensor [*, featureSize]
 * \param indices Pointer to indices array [numElements]
 * \param numElements Number of elements to gather
 * \param featureSize Size of each feature vector
 * \param stream CUDA stream for asynchronous execution
 */
template <typename T>
void invokeMoeGather(
    T* output, T const* input, int const* indices, int numElements, int featureSize, cudaStream_t stream);

/*!
 * \brief Scatter elements to output tensor according to indices using optimized CUDA kernels
 *
 * This function performs a scatter operation where elements from the input tensor are distributed
 * to the output tensor according to the specified indices. The operation is optimized
 * with 128-bit vectorized memory operations when possible.
 *
 * The operation is equivalent to:
 * for (int i = 0; i < numElements; i++) {
 *     int dstIdx = indices[i];
 *     if (dstIdx >= 0) {
 *         output[dstIdx] = input[i];  // Copy entire feature vector
 *     }
 * }
 *
 * Supports various data types including:
 * - float (32-bit): 4 elements per 128-bit vector
 * - half (16-bit): 8 elements per 128-bit vector
 * - bfloat16 (16-bit): 8 elements per 128-bit vector
 * - FP8 types (8-bit): 16 elements per 128-bit vector
 * - NVFP4 types (4-bit, packed): 32 elements per 128-bit vector
 *
 * \param output Pointer to output tensor [*, featureSize]
 * \param input Pointer to input tensor [numElements, featureSize]
 * \param indices Pointer to indices array [numElements]
 * \param numElements Number of elements to scatter
 * \param featureSize Size of each feature vector
 * \param stream CUDA stream for asynchronous execution
 */
template <typename T>
void invokeMoeScatter(
    T* output, T const* input, int const* indices, int numElements, int featureSize, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
