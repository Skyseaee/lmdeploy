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

#include "src/turbomind/core/quant_mode.h"
#include "src/turbomind/kernels/cutlass_kernels/include/moe_kernels.h"
#include "src/turbomind/kernels/gemm_profiler/gemmPluginProfiler.h"

//namespace tkc = tensorrt_llm::kernels::cutlass_kernels;

namespace tensorrt_llm::plugins
{

using CutlassMoeFCRunnerPtr = std::shared_ptr<tensorrt_llm::kernels::cutlass_kernels::CutlassMoeFCRunnerInterface>;
using MOEParallelismConfig = tensorrt_llm::kernels::cutlass_kernels::MOEParallelismConfig;
//using MOEExpertScaleNormalizationMode = tensorrt_llm_moe_fp8::kernels::MOEExpertScaleNormalizationMode;

class MixtureOfExpertsGemmProfiler : public GemmPluginProfiler<tensorrt_llm::cutlass_extensions::CutlassGemmConfig,
                                     CutlassMoeFCRunnerPtr, GemmIDMoe, GemmIDMoeHash>

{
    public:
        using Config = tensorrt_llm::cutlass_extensions::CutlassGemmConfig;

        void setQuantMode(turbomind::QuantMode const& quantMode);

        void setMaxProfileM(int maxProfileM)
        {
            mMaxProfileM = maxProfileM;
        }

        virtual int getMaxProfileM() const override;

	void setMoEParam(const int                       expert_num,
                         const int                       experts_per_token,
                         const int                       expert_hidden_dim,
                         const int                       expert_inter_size,
                         tensorrt_llm::kernels::cutlass_kernels::ActivationType    act_type,
                         //MOEExpertScaleNormalizationMode normal_type,
                         MOEParallelismConfig            paral_config,
                         turbomind::DataType             dtype,
                         turbomind::DataType             wtype,
                         turbomind::DataType             otype);
	
	void setGemmToProfile(tensorrt_llm::kernels::cutlass_kernels::GemmProfilerBackend::GemmToProfile gemm_to_profile);

    protected:
        void runTactic(int m, int n, int k, Config const &tactic, char *workspace, cudaStream_t const &stream) override;

        void computeTmpSize(size_t maxM, size_t n, size_t k) override;

        // TODO(anchengc) implement checkTactic
        // bool checkTactic(int m, int n, int k, const Config& tactic) const override;

        std::vector<Config> getTactics(int m, int n, int k) const override;

        void initTmpData(int m, int n, int k, char *workspace, size_t size, cudaStream_t stream) override;

        void checkInit();

        bool init_backend = false;
        tensorrt_llm::kernels::cutlass_kernels::GemmProfilerBackend backend{};

    private:
        size_t getBytePerElement(turbomind::DataType type);

        turbomind::QuantMode mQuantMode = turbomind::QuantMode::fromDescription();

        turbomind::DataType mType = turbomind::DataType::kFloat8_e4m3;

        // MoE param
        int m_expert_num;
        int m_experts_per_token;
        int m_expert_hidden_dim;
        int m_expert_inter_size;
        tensorrt_llm::kernels::cutlass_kernels::ActivationType m_act_type;
        //MOEExpertScaleNormalizationMode m_normal_type;
        MOEParallelismConfig m_paral_config;
        turbomind::DataType m_dtype;
        turbomind::DataType m_wtype;
        turbomind::DataType m_otype;

	int mMaxProfileM = 0;
};

} // namespace tensorrt_llm::plugins
