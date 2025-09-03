/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Modified from https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/layers/DenseWeight.h

#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/core/quant_mode.h"

#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

inline LoraPolicy getLoraPolicy(const std::string& policy)
{
    if (policy == "plora") {
        return LoraPolicy::kPlora;
    }
    return LoraPolicy::kNull;
}

struct LoraWeight {
    LoraPolicy policy;
    int        r = 0;
    float      scale;
    void*      a = nullptr;
    void*      b = nullptr;
};

struct LlamaDenseWeight: public core::Module {

    LlamaDenseWeight(): data_type{}, weight_type{}, lora{}, k_desc{}, q_desc{} {}

    // Note(meng): expert_num: Support the per-expert-quant method
    // Note(meng): If a non-1 value is passed in, it means that there are expert_num weight_scale, d0_scale and d1_scale values.
    void emplace(int input_dim, int output_dim, DataType data_type, bool bias, DataType weight_type, int group_size, QuantMode quant_mode, int expert_num = 1);

    void prepare(bool fused_moe, bool use_simt);

    LlamaDenseWeight& operator=(std::nullptr_t)
    {
        this->~LlamaDenseWeight();
        new (this) LlamaDenseWeight{};
        return *this;
    }

    operator bool() const noexcept
    {
        return static_cast<bool>(weight);
    }

    int input_dim  = 0;
    int output_dim = 0;
    int group_size = 1;

    DataType data_type;
    DataType weight_type;

    Tensor weight;
    Tensor bias;

    Tensor scales;
    Tensor zeros;

    Tensor scales_zeros;

    LoraWeight lora;

    gemm::MatrixLayout k_desc;
    gemm::MatrixLayout q_desc;

    QuantMode quant_mode = QuantMode::fromDescription();

#ifdef ENABLE_FP8
    // NOTE: FP8 scales
    // scale = AMAX(tensor) / FP8_MAX
    // During GEMM, A (original) = A_scaled (fp8) * "scale of A"
    // float* input_scale      = nullptr;  // a scalar
    // float* input_scale_inv  = nullptr;  // a scalar inv
    // float* weight_scale     = nullptr;  // a scaler or a vector
    // float* weight_scale_inv = nullptr;

    // // NOTE: used for fused gemm
    // float* d0_scale             = nullptr;  // a scalar
    // float* d1_scale             = nullptr;  // a scalar
    // float  host_d0_scale        = 1.0f;     // a scalar
    // float  host_d1_scale        = 1.0f;     // a scalar
    // float  host_input_scale_inv = 1.0f;     // a scalar

    Tensor input_scale;      // a scalar
    Tensor input_scale_inv;  // a scalar inv
    Tensor weight_scale;     // a scaler or a vector
    Tensor weight_scale_inv;

    // NOTE: used for fused gemm
    Tensor d0_scale;              // a scalar
    Tensor d1_scale;              // a scalar
    Tensor host_d0_scale;         // a scalar
    Tensor host_d1_scale;         // a scalar
    Tensor host_input_scale_inv;  // a scalar

    Tensor pre_quant_scale;      // per channel quant scale
    Tensor w_group_quant_scale;  // per group quant scale
#endif
};

struct LlamaAttentionWeight: public core::Module {

    LlamaAttentionWeight() = default;

    LlamaAttentionWeight(int       hidden_dim,
                         int       head_dim,
                         int       head_num,
                         int       kv_head_num,
                         MLAParam  mla,
                         bool      bias,
                         bool      qk_norm,
                         int       tp_size,
                         int       tp_rank,
                         DataType  data_type,
                         DataType  weight_type,
                         int       group_siz,
                         QuantMode quant_mode);

    void prepare(bool use_simt);

    LlamaDenseWeight qkv;
    LlamaDenseWeight output;

    LlamaDenseWeight q_proj;
    LlamaDenseWeight q_a_proj;
    LlamaDenseWeight q_b_proj;
    LlamaDenseWeight kv_a_proj;
    LlamaDenseWeight kv_b_proj;

    Tensor q_a_layernorm;
    Tensor kv_a_layernorm;

    QuantMode quant_mode{};
};

struct LlamaFfnWeight: core::Module {

    LlamaFfnWeight() = default;

    LlamaFfnWeight(int       hidden_dim,
                   int       inter_size,
                   int       tp_size,
                   int       tp_rank,
                   DataType  data_type,
                   DataType  weight_type,
                   int       group_size,
                   bool      fuse_silu_act,
                   QuantMode quant_mode);

    static constexpr bool fuse_up_and_gate = true;

    void prepare(bool fused_moe, bool use_simt);

    LlamaDenseWeight gating;
    LlamaDenseWeight intermediate;
    LlamaDenseWeight output;
    LlamaDenseWeight fused_gating_intermediate;

    int  inter_size{};
    bool is_fused_silu{};
    QuantMode quant_mode{};
};

struct MoeFfnWeight: core::Module {

    MoeFfnWeight() = default;

    MoeFfnWeight(int             layer_id,
                 const MoeParam& param,
                 int             hidden_dim,
                 DataType        data_type,
                 DataType        weight_type,
                 int             group_size,
                 int             tp_size,
                 int             tp_rank,
                 int             ep_size,
                 int             ep_rank,
                 bool            fuse_silu_act,
                 QuantMode       quant_mode,
                 bool            cutlass_fused_kernel = false);

    void prepare(bool use_simt);

    void process_fp8_moe_weight();

    bool same_config(const LlamaFfnWeight& e1, const LlamaFfnWeight& e2){
        return ((e1.gating.input_dim==e2.gating.input_dim) && (e1.gating.output_dim==e2.gating.output_dim) &&
            (e1.gating.group_size==e2.gating.group_size) && (e1.gating.data_type==e2.gating.data_type) && (e1.gating.weight_type==e2.gating.weight_type)) &&
            ((e1.intermediate.input_dim==e2.intermediate.input_dim) && (e1.intermediate.output_dim==e2.intermediate.output_dim) &&
            (e1.intermediate.group_size==e2.intermediate.group_size) && (e1.intermediate.data_type==e2.intermediate.data_type) && (e1.intermediate.weight_type==e2.intermediate.weight_type)) &&
            ((e1.output.input_dim==e2.output.input_dim) && (e1.output.output_dim==e2.output.output_dim) &&
            (e1.output.group_size==e2.output.group_size) && (e1.output.data_type==e2.output.data_type) && (e1.output.weight_type==e2.output.weight_type));
    }

    LlamaDenseWeight gate;
    LlamaDenseWeight shared_gate;

    LlamaFfnWeight fused_expert;

    std::vector<std::unique_ptr<LlamaFfnWeight>> experts;

    // reference into `experts`
    LlamaFfnWeight block;

    MoeParam::Method method{};
    QuantMode        quant_mode{};
};

}  // namespace turbomind
