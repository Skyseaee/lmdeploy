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
#include "src/turbomind/kernels/gemm_profiler/gemmMoEProfiler.h"

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;
using tensorrt_llm::plugins::MixtureOfExpertsGemmProfiler;

size_t MixtureOfExpertsGemmProfiler::getBytePerElement(turbomind::DataType type)
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

void MixtureOfExpertsGemmProfiler::setQuantMode(tensorrt_llm::common::QuantMode const &quantMode)
{
    mQuantMode = quantMode;
}

void MixtureOfExpertsGemmProfiler::runTactic(
    int m, int n, int k, MixtureOfExpertsGemmProfiler::Config const &tactic, char *workspace, cudaStream_t const &stream)
{
    checkInit();
    backend.runProfiler(m, tactic, workspace, stream);

    // size_t bpe = getBytePerElement(mType);

    // // Workspace size required by gemm runner
    // // NB: this function will throw exception when selected tactic exceeds SMEM, which is then
    // // caught by gemmPluginProfiler and it will register this tactic as invalid
    // size_t wsSizeRunner = mRunner->getWorkspaceSize(m, n, k,
    //                                                 m_expert_num, m_experts_per_token, m_act_type, m_normal_type, m_paral_config, /*hasLora()*/ false);

    // // Workspace size required by profiling
    // size_t wsByteOffset = 0;
    // int8_t *wsBytePointer = reinterpret_cast<int8_t *>(workspace);
    // void *aTmp = reinterpret_cast<void *>(nextWorkspacePtr(wsBytePointer, wsByteOffset, m * k * bpe));   // input
    // void *logits = reinterpret_cast<void *>(nextWorkspacePtr(wsBytePointer, wsByteOffset, m * m_expert_num * getBytePerElement(turbomind::DataType::TYPE_FP32)));   // input
    // void *bTmp = reinterpret_cast<void *>(nextWorkspacePtr(wsBytePointer, wsByteOffset, 2 * n * k * m_expert_num * bpe)); // w1w3
    // void *cTmp = reinterpret_cast<void *>(nextWorkspacePtr(wsBytePointer, wsByteOffset, n * k * m_expert_num * bpe));     // w2
    // void *dTmp = reinterpret_cast<void *>(nextWorkspacePtr(wsBytePointer, wsByteOffset, m * n * bpe));
    // void *scales = reinterpret_cast<void *>(nextWorkspacePtr(wsBytePointer, wsByteOffset, m * m_expert_num * getBytePerElement(turbomind::DataType::TYPE_FP32)));
    // void *map1 = reinterpret_cast<void *>(nextWorkspacePtr(wsBytePointer, wsByteOffset, m * m_experts_per_token * getBytePerElement(turbomind::DataType::TYPE_INT8)));
    // void *map2 = reinterpret_cast<void *>(nextWorkspacePtr(wsBytePointer, wsByteOffset, m * m_expert_num * getBytePerElement(turbomind::DataType::TYPE_INT8)));
    // char *workspaceTmp = reinterpret_cast<char *>(nextWorkspacePtr(wsBytePointer, wsByteOffset, wsSizeRunner));

    // tensorrt_llm::kernels::LoraParams lora_params{};

    // // FP16 Quant Param
    // tensorrt_llm::kernels::QuantParams quant_params{};

    // // Run profiling
    // mRunner->runMoe(
    //     /*void const**/ aTmp,
    //     /*float const* */ logits, // [tokens * param_.expert_num]
    //     /*void const* */ bTmp,    // fc1_expert_weights_void
    //     /*void const* */ nullptr, // fc1_expert_biases_void
    //     /*ActivationType */ m_act_type,
    //     /*void const* */ cTmp,    // fc2_expert_weights_void
    //     /*void const* */ nullptr, // fc2_expert_biases_void
    //     /*QuantParams */ quant_params,
    //     /*int64_t const */ m,
    //     /*int64_t const */ k,
    //     /*int64_t const */ n,
    //     /*int const */ m_expert_num,
    //     /*int const */ m_experts_per_token,
    //     /*char* */ static_cast<char *>(workspaceTmp),

    //     // Outputs
    //     /*void* */ dTmp,
    //     /*bool const* */ nullptr, // finished, (is deprecated)
    //     /*int64_t const */ m,     // active_rows, (is deprecated)
    //     /*void* */ scales,       // token_topk_unpermuted_scales, m_workspace.scale_probs,
    //     /*int* */ map1,        // expanded_source_row_to_expanded_dest_row, static_cast<int *>(m_workspace.src_to_dest_map),
    //     /*int* */ map2,        // expert_for_source_row,  static_cast<int *>(m_workspace.selected_experts)
    //     /*float */ -INFINITY,     // sparse_mixer_epsilon must be set when normalization mode is SPARSE_MIXER mSparseMixerEpsilon
    //     /*MOEParallelismConfig */ m_paral_config,
    //     /*MOEExpertScaleNormalizationMode */ m_normal_type,
    //     /*bool */ false, // hasLora() = false, now we cant support lora in Moe
    //     /*LoraParams& */ lora_params,
    //     /*cudaStream_t */ stream);
}

int MixtureOfExpertsGemmProfiler::getMaxProfileM() const
{
	return mMaxProfileM;
}

void MixtureOfExpertsGemmProfiler::computeTmpSize(size_t maxM, size_t n, size_t k)
{
    checkInit();
    size_t bytes = backend.getWorkspaceSize(maxM);
    this->setTmpWorkspaceSizeInBytes(bytes);

    // std::vector<size_t> workspaces{
    //     moe_workspace_size,
    //     scale_probabilities_size,
    //     src_to_dest_map_size,
    //     selected_expert_size,
    //     lora_workspace_size,
    // };

    // // no bias & no lora
    // std::vector<size_t> workspaces{
    //     maxM * k * getBytePerElement(mType),                 // A input [token_num * hidden_dim]
    //     maxM * m_expert_num * getBytePerElement(float),                 // logits input [token_num * expert_num]
    //     2 * n * k * m_expert_num * getBytePerElement(mType), // w1w3 [hidden_dim * inter_size * 2 * expert_num]
    //     n * k * m_expert_num * getBytePerElement(mType),     // w2 [hidden_dim * inter_size * expert_num]
    //     maxM * n * getBytePerElement(mType),                 // output [token_num * hidden_dim]
    //     maxM * m_expert_num * sizeof(float),                 // Output of post-softmax routing probabilities
    //     maxM * m_experts_per_token * sizeof(int),            // Permutation map
    //     maxM * m_experts_per_token * sizeof(int),            // Selected expert map
    //     mRunner->getWorkspaceSize(maxM, n, k,
    //                               m_expert_num, m_experts_per_token, m_act_type, m_normal_type, m_paral_config, /*hasLora()*/ false), // workspace
    // };
    // size_t bytes = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
    // setTmpWorkspaceSizeInBytes(bytes);
}

std::vector<MixtureOfExpertsGemmProfiler::Config> MixtureOfExpertsGemmProfiler::getTactics(int m, int n, int k) const
{
    return mRunner->getTactics();
}

void MixtureOfExpertsGemmProfiler::setMoEParam(const int expert_num, const int experts_per_token, const int expert_hidden_dim, const int expert_inter_size,
    tensorrt_llm::ActivationType act_type, MOEExpertScaleNormalizationMode normal_type, MOEParallelismConfig paral_config,
    turbomind::DataType dtype, turbomind::DataType wtype, turbomind::DataType otype)
{
    // MoE param
    m_expert_num = expert_num;
    m_experts_per_token = experts_per_token;
    m_act_type = act_type;
    m_normal_type = normal_type;
    m_paral_config = paral_config;
    m_dtype = dtype;
    m_wtype = wtype;
    m_otype = otype;

    m_expert_hidden_dim = expert_hidden_dim;
    m_expert_inter_size = expert_inter_size;
}

void MixtureOfExpertsGemmProfiler::checkInit()
{
    //assert(mRunner);
    if (init_backend)
    {
        return;
    }
    init_backend = true;
    backend.init(*mRunner.get(), backend.mGemmToProfile, m_dtype, m_wtype, m_otype,
        m_expert_num, m_experts_per_token, m_expert_hidden_dim, m_expert_inter_size, m_act_type,
        false, false, m_paral_config);
}

void MixtureOfExpertsGemmProfiler::setGemmToProfile(tensorrt_llm::kernels::GemmProfilerBackend::GemmToProfile gemm_to_profile)
{
    // Just set the backend directly. This will just be reused in checkInit().
    backend.mGemmToProfile = gemm_to_profile;
    // We need to set the backend to reinitialise itself with the new GEMM
    init_backend = false;
}
