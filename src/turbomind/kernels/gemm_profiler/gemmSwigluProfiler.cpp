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

#include "src/turbomind/kernels/fused_gated_gemm/gemm_configs.h"
#include "src/turbomind/kernels/gemm_profiler/gemmSwigluProfiler.h"

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels::cutlass_kernels;
using tensorrt_llm::plugins::GemmSwigluPluginProfiler;

size_t GemmSwigluPluginProfiler::getBytePerElement(turbomind::DataType type)
{
    size_t bpe;
    if (type == turbomind::DataType::TYPE_FP16 || type == turbomind::DataType::TYPE_BF16)
    {
        bpe = 2;
    }
    else if (type == turbomind::DataType::TYPE_INT8 || type == turbomind::DataType::TYPE_FP8_E4M3)
    {
        bpe = 1;
    }
    else
    {
        TM_LOG_ERROR("Not recognized/implemented");
    }
    return bpe;
}

void GemmSwigluPluginProfiler::setQuantMode(tensorrt_llm::common::QuantMode const& quantMode)
{
    mQuantMode = quantMode;
}

void GemmSwigluPluginProfiler::runTactic(
    int m, int n, int k, GemmSwigluPluginProfiler::Config const& tactic, char* workspace, cudaStream_t const& stream)
{
    size_t bpe = getBytePerElement(mType);

    // Workspace size required by gemm runner
    // NB: this function will throw exception when selected tactic exceeds SMEM, which is then
    // caught by gemmPluginProfiler and it will register this tactic as invalid
    size_t wsSizeRunner = mRunner->getWorkspaceSize(m, n, k);

    // Workspace size required by profiling
    size_t wsByteOffset = 0;
    int8_t* wsBytePointer = reinterpret_cast<int8_t*>(workspace);
    void* aTmp = reinterpret_cast<void*>(nextWorkspacePtr(wsBytePointer, wsByteOffset, m * k * bpe));
    void* bTmp = reinterpret_cast<void*>(nextWorkspacePtr(wsBytePointer, wsByteOffset, n * k * bpe));
    void* cTmp = reinterpret_cast<void*>(nextWorkspacePtr(wsBytePointer, wsByteOffset, 1 * n * bpe));
    void* dTmp = reinterpret_cast<void*>(nextWorkspacePtr(wsBytePointer, wsByteOffset, m * (n / 2) * bpe));
    char* workspaceTmp = reinterpret_cast<char*>(nextWorkspacePtr(wsBytePointer, wsByteOffset, wsSizeRunner));

    // Run profiling
    mRunner->gemm(
        dTmp, aTmp, bTmp, cTmp, mQuantMode, m, n, k, 1.0, 1.0, 1.0, tactic, workspaceTmp, wsSizeRunner, stream);
}

int GemmSwigluPluginProfiler::getMaxProfileM() const
{
    return 32768;
}

void GemmSwigluPluginProfiler::computeTmpSize(size_t maxM, size_t n, size_t k)
{
    std::vector<size_t> workspaces = {
        maxM * k * getBytePerElement(mType),       // A
        n * k * getBytePerElement(mType),          // B
        1 * n * getBytePerElement(mType),          // C_bias
        maxM * (n / 2) * getBytePerElement(mType), // D
        mRunner->getWorkspaceSize(maxM, n, k)      // workspace
    };
    size_t bytes = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
    setTmpWorkspaceSizeInBytes(bytes);
}

std::vector<GemmSwigluPluginProfiler::Config> GemmSwigluPluginProfiler::getTactics(int m, int n, int k) const
{
    return mRunner->getConfigs();
}
// IPluginV2 Methods