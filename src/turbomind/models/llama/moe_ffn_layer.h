// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/context.h"
#include "src/turbomind/kernels/gemm/moe_utils_v2.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/LlamaFfnLayer.h"
#include "src/turbomind/models/llama/llama_params.h"

#ifdef FUSED_MOE_FFN_GEMM
#include "src/turbomind/kernels/cutlass_kernels/include/moe_kernels.h"
#include "src/turbomind/kernels/gemm_profiler/gemmMoEProfiler.h"

namespace tlp = tensorrt_llm::plugins;
namespace tkc = tensorrt_llm::cutlass_extensions;
#endif

namespace turbomind {

#ifdef FUSED_MOE_FFN_GEMM
using MoEGemmSwigluRunnerPtr = std::shared_ptr<tensorrt_llm::kernels::cutlass_kernels::CutlassMoeFCRunnerInterface>;
using MOEParallelismConfig   = tensorrt_llm::kernels::cutlass_kernels::MOEParallelismConfig;
// using MOEExpertScaleNormalizationMode = tensorrt_llm::kernels::cutlass_kernels::MOEExpertScaleNormalizationMode;

using MixtureOfExpertsGemmProfilerPtr = std::shared_ptr<tensorrt_llm::plugins::MixtureOfExpertsGemmProfiler>;
using MOEGemmPluginProfilerManager
    = tensorrt_llm::plugins::GemmPluginProfilerManager<tensorrt_llm::plugins::MixtureOfExpertsGemmProfiler>;
#endif

class MoeFfnLayer {
public:
    MoeFfnLayer(const ModelParam& model, const MoeParam& param, const EngineParam& engine, const Context& ctx);

    struct ForwardParam {
        Tensor              input;
        Tensor              output;
        Tensor              moe_fp16_buf;
        Tensor              gate_fp32_buf;
        const MoeFfnWeight* weights;
        float               scale;
        int                 layer_id;

        int          pf_batch_size        = 0;
        float        output_scale         = 1.0f;
        //Tensor       inter_buf_fp8{};
        bool         use_shared_stream    = false;
        cudaEvent_t  shared_expert_event  = nullptr;
        cudaStream_t shared_expert_stream = nullptr;
    };

    void Forward(ForwardParam& p);

    void Combine(ForwardParam& p);

private:
    Tensor_<float> Gate(const Tensor& input, const LlamaDenseWeight& gate);
    void Gate(const Tensor& input, const LlamaDenseWeight& gate, Tensor& output);

    void dump_logits(int token_num, int layer_id, int expert_num);

    const int      inter_size_;
    const int      moe_ep_size_;
    const int      moe_ep_rank_;
    const int      hidden_dim_;
    const MoeParam param_;
    const QuantMode quant_mode_;

    cudaStream_t const stream_;
    LlamaLinear&       linear_;

    std::unique_ptr<LlamaFfnLayer>        expert_ffn_;
    std::unique_ptr<gemm::MoeGemmContext> context_;

    ///////////////////////////////////////////////////////
    /// runtime states
    Buffer_<int> h_offsets_;

    Buffer_<int>   masks_;
    Buffer_<int>   f2n_;
    Buffer_<int>   en2f_;
    Buffer_<float> scales_;
    Buffer_<int>   accum_;
    Buffer_<int>   offsets_;

    Tensor         temp_;
    Tensor_<float> shared_scales_;
    ///////////////////////////////////////////////////////
    int2 expert_range_{};

    int mlp_tp_size_ = 1;
    int mlp_tp_rank_ = 0;

    Buffer_<int> votes_{};
    Buffer_<int> hists_{};
    Tensor       cutlass_inout_buf_;

    std::vector<int> shared_expert_inter_size_;
};

}  // namespace turbomind
