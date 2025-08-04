/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cassert>
#include <set>
#include <string>
#include <vector>

#include "src/turbomind/kernels/fused_gated_gemm/fused_gated_gemm.h"
#include "src/turbomind/kernels/gemm_profiler/gemmPluginProfiler.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace tensorrt_llm::plugins
{

using GemmSwigluRunnerPtr
    = std::shared_ptr<tensorrt_llm::kernels::cutlass_kernels::CutlassFusedGatedGemmRunnerInterface>;

class GemmSwigluPluginProfiler : public GemmPluginProfiler<tensorrt_llm::cutlass_extensions::CutlassGemmConfig,
                                     GemmSwigluRunnerPtr, GemmIdCore, GemmIdCoreHash>

{
public:
    using Config = tensorrt_llm::cutlass_extensions::CutlassGemmConfig;

    void setQuantMode(turbomind::QuantMode const& quantMode);

    virtual int getMaxProfileM() const override;

protected:
    void runTactic(int m, int n, int k, Config const& tactic, char* workspace, cudaStream_t const& stream) override;

    void computeTmpSize(size_t maxM, size_t n, size_t k) override;

    // TODO(anchengc) implement checkTactic
    // bool checkTactic(int m, int n, int k, const Config& tactic) const override;

    std::vector<Config> getTactics(int m, int n, int k) const override;

    void initTmpData(int m, int n, int k, char* workspace, size_t size, cudaStream_t stream) override;

private:
    size_t getBytePerElement(turbomind::DataType type);

    turbomind::QuantMode mQuantMode = turbomind::QuantMode::fromDescription();

    turbomind::DataType mType = turbomind::DataType::kFloat8_e4m3;
};

} // namespace tensorrt_llm::plugins
