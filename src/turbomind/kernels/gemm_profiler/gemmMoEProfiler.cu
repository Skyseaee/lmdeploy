/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include <numeric>
#include <cutlass/util/reference/device/tensor_fill.h>

#include "src/turbomind/kernels/cutlass_extensions/include/cutlass_extensions/gemm_configs.h"
#include "src/turbomind/kernels/gemm_profiler/gemmMoEProfiler.h"

using tensorrt_llm::plugins::MixtureOfExpertsGemmProfiler;

void MixtureOfExpertsGemmProfiler::initTmpData(int m, int n, int k, char* workspace, size_t size, cudaStream_t stream)
{
    size_t bpe = getBytePerElement(mType);

    if (mType == turbomind::DataType::kFloat8_e4m3)
    {
        // set random data to input data
        // just input and weight
        cutlass::reference::device::BlockFillRandomUniform(reinterpret_cast<cutlass::float_e4m3_t*>(workspace),
            m * k + 2 * n * k * m_expert_num + n * k * m_expert_num, 42, cutlass::float_e4m3_t{128}, -cutlass::float_e4m3_t{128}, -1, 0, stream);
    }
}
