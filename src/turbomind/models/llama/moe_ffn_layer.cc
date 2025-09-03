// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

#include "src/turbomind/kernels/activation_kernels.h"

#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/moe_ffn_layer.h"

#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

MoeFfnLayer::MoeFfnLayer(const ModelParam& model, const MoeParam& param, const EngineParam& engine, const Context& ctx):
    inter_size_(param.inter_size / engine.moe_tp_size),
    moe_ep_size_(engine.moe_ep_size),
    moe_ep_rank_(engine.moe_ep_rank),
    hidden_dim_(model.hidden_units),
    param_(param),
    stream_(ctx.stream),
    linear_(*ctx.linear),
    quant_mode_(model.quant_mode),
    mlp_tp_size_(engine.mlp_tp_size),
    mlp_tp_rank_(engine.mlp_tp_rank),
    shared_expert_inter_size_(model.inter_size)
{
    TM_CHECK(!param.expert_num.empty());

    const int max_expert_num = engine.moe_ep_size * (int)*std::max_element(param.expert_num.begin(), param.expert_num.end());

    if (param_.method == MoeParam::kFused) {
        context_ =
            std::make_unique<gemm::MoeGemmContext>(max_expert_num, param.experts_per_token, ctx.device_prop, stream_);
    }
    else {
        expert_ffn_ = std::make_unique<LlamaFfnLayer>(model, ctx);
    }

    h_offsets_ = {max_expert_num + 1, kCPUpinned};

    const int max_token_num = engine.max_forward_token_num;
    const int pad_token_num = (max_token_num + kMoeGateVecSize - 1) / kMoeGateVecSize * kMoeGateVecSize;

    masks_   = {max_expert_num * pad_token_num, kDEVICE};
    f2n_     = {param_.experts_per_token * max_token_num, kDEVICE};
    en2f_    = {param_.experts_per_token * max_token_num, kDEVICE};
    scales_  = {param_.experts_per_token * max_token_num, kDEVICE};
    offsets_ = {max_expert_num + 1, kDEVICE};
    accum_   = {max_expert_num * kMoeGateMaxTiles, kDEVICE};

    votes_ = {max_expert_num, kDEVICE};
    hists_ = {engine.max_batch_size + 1, kDEVICE};
}

void MoeFfnLayer::Gate(const Tensor& input, const LlamaDenseWeight& gate, Tensor& output)
{
    auto& weight = gate.weight;
    TM_CHECK_EQ(input.shape(1), weight.shape(0));
    linear_.forward(input, gate, LlamaLinear::kGemm, output);
    sync_check_cuda_error();
}

Tensor_<float> MoeFfnLayer::Gate(const Tensor& input, const LlamaDenseWeight& gate)
{
    auto& weight = gate.weight;
    TM_CHECK_EQ(input.shape(1), weight.shape(0));
    Tensor_<float> logits{{input.shape(0), weight.shape(1)}, kDEVICE};
    linear_.forward(input, gate, LlamaLinear::kGemm, logits);
    sync_check_cuda_error();
    return logits;
}

void MoeFfnLayer::Forward(ForwardParam& p)
{
    const int tokens = p.input.shape(0);

    bool enable_expert_pruning = param_.enable_expert_pruning && p.pf_batch_size && tokens >= 16;

    const auto& moe = *p.weights;

    const size_t padded = (tokens + kMoeGateVecSize - 1) / kMoeGateVecSize * kMoeGateVecSize;

    const int    local_expert_num = moe.experts.size();
    const int    expert_num       = local_expert_num * moe_ep_size_;
    expert_range_ = {moe_ep_rank_ * local_expert_num, (moe_ep_rank_+1) * local_expert_num};

    FT_CHECK(expert_num);

    auto logits = p.gate_fp32_buf.slice(0, tokens);
    Gate(p.moe_fp16_buf, moe.gate, logits);
    //auto logits = Gate(p.moe_fp16_buf, moe.gate);

    if (enable_expert_pruning) {
        // only support with expert=48 && decoding stage && batchsize=[16, 256]
        invokeMaskExpertsByVoteFusedV2(static_cast<float*>(logits.raw_data()),
                                       votes_.data(),
                                       hists_.data(),
                                       tokens,
                                       expert_num,
                                       param_.experts_per_token,
                                       param_.keep_expert_num,
                                       stream_);
        sync_check_cuda_error();
    }

    // clang-format off
#ifdef FUSED_MOE_FFN_GEMM
    if (quant_mode_.isFP8Static())
    {
        cutlass_inout_buf_ = Tensor{{param_.experts_per_token * tokens, hidden_dim_}, p.output.dtype(), p.output.device()};

        linear_.forward_cutlass_moe(p.output,
                                    p.input,
                                    logits, 
                                    //p.inter_buf_fp8,
                                    cutlass_inout_buf_,
                                    moe.fused_expert,
                                    tokens,
                                    expert_num,
                                    p.use_shared_stream,
                                    p.shared_expert_event,
                                    p.shared_expert_stream);
    }
    else
#endif 
    {
        // dump_logits(tokens, layer_id);
        check_cuda_error(cudaMemsetAsync(accum_.data(), 0, sizeof(int) * expert_num * kMoeGateMaxTiles, stream_));
        check_cuda_error(cudaMemsetAsync(masks_.data(), -1, sizeof(int8_t) * expert_num * padded, stream_));

        bool softmax = true;
        if (param_.topk_method == "group_limited_greedy") {
            invokeMoeSoftmaxMaskTopKGroups(
            static_cast<float*>(logits.raw_data()), tokens, expert_num, expert_num / param_.n_group, param_.topk_group, stream_);
            sync_check_cuda_error();
            softmax = false;
        }

        // dump_logits(tokens, layer_id);
        /// TODO: fix illegal memory access even if NaN are present in logits
        invokeMoeGate_V2(f2n_.data(),
                         en2f_.data(),
                         offsets_.data(),
                         scales_.data(),
                         masks_.data(),
                         accum_.data(),
                         static_cast<float*>(logits.raw_data()),
                         tokens,
                         padded,
                         expert_num,
                         param_.experts_per_token,
                         softmax,
                         param_.norm_topk_prob,
                         param_.routed_scale,
                         expert_range_,
                         stream_);
        sync_check_cuda_error();

        if (isTuning()) {
            std::mt19937     g;
            const auto       expert_ids = SampleUniform(tokens, expert_num, param_.experts_per_token, g);
            std::vector<int> cnt(expert_num);
            for (const auto& x : expert_ids) {
                ++cnt[x];
            }
            h_offsets_[0] = 0;
            for (int i = 0; i < expert_num; ++i) {
                h_offsets_[i + 1] = h_offsets_[i] + cnt[i];
            }
            check_cuda_error(cudaMemcpyAsync(
                offsets_.data(), h_offsets_.data(), sizeof(int) * (expert_num + 1), cudaMemcpyDefault, stream_));
        }

        if (moe_ep_size_ > 1) {
            invokeMoveOffsets(offsets_.data(), expert_num, expert_range_, stream_);
        }

        temp_ = Tensor{{param_.experts_per_token * tokens, hidden_dim_}, p.input.dtype(), p.input.device()};

        if (moe_ep_size_ > 1) {
            // initializing moe output with zeros, preventing nan in unused memory
            // only observed when ep > 1
            auto set_to_zeros = [&](auto t) {
                using T = decltype(t);
                check_cuda_error(cudaMemsetAsync(temp_.data<T>(), 0, sizeof(T) * temp_.size(), stream_));
            };
            TM_DISPATCH_DTYPES(temp_.dtype(), set_to_zeros, float, half_t, bfloat16_t);
        }

        if (param_.method == MoeParam::kNaive) {

            invokeMoeDispatch(temp_, p.input, f2n_.data(), param_.experts_per_token, stream_);
            sync_check_cuda_error();

            check_cuda_error(cudaMemcpyAsync(
                h_offsets_.data(), offsets_.data(), sizeof(int) * (local_expert_num + 1), cudaMemcpyDefault, stream_));

            check_cuda_error(cudaStreamSynchronize(stream_));

            TM_CHECK_EQ(h_offsets_[local_expert_num], tokens * param_.experts_per_token);

            for (int i = 0; i < local_expert_num; ++i) {
                if (int count = h_offsets_[i + 1] - h_offsets_[i]) {
                    auto io = temp_.slice({h_offsets_[i], 0}, {count, -1});
                    expert_ffn_->forward({io, io, moe.experts.at(i).get(), p.layer_id});
                }
            }
        }
        else {
            context_->update(local_expert_num, param_.experts_per_token, offsets_.data());

            auto& block = moe.block;

            const int inter_dim = block.is_fused_silu ? inter_size_ : inter_size_ * 2;
            Tensor    inter{{tokens * param_.experts_per_token, inter_dim}, p.input.dtype(), p.input.device()};

            linear_.forward_moe(inter,
                                p.input,
                                f2n_.data(),
                                offsets_.data(),
                                block.fused_gating_intermediate,
                                block.is_fused_silu ? LlamaLinear::kFusedSiluFfn : LlamaLinear::kGemm,
                                context_.get());
            sync_check_cuda_error();

            if (!block.is_fused_silu) {
                invokeGenericActivation_v3<SiluActivation>(inter.slice({0, 0}, {-1, inter_size_}),  //
                                                           inter.slice({0, inter_size_}, {-1, -1}),
                                                           stream_);
                sync_check_cuda_error();
            }

            linear_.forward_moe(temp_,
                                inter.slice({0, 0}, {-1, inter_size_}),
                                nullptr,
                                offsets_.data(),
                                block.output,
                                LlamaLinear::kGemm,
                                context_.get());
            sync_check_cuda_error();
        }
    }
    // clang-format on

    if (moe.shared_gate.weight) {
        shared_scales_ = Gate(p.input, moe.shared_gate);
    }
}

void MoeFfnLayer::Combine(ForwardParam& p)
{
    auto& moe = *p.weights;
    const int tokens = p.output.shape(0);
    // clang-format off
#ifdef FUSED_MOE_FFN_GEMM
    if (quant_mode_.isFP8Static()) 
    {
        if (shared_expert_inter_size_[p.layer_id] == 0) 
        {
            // don't have shared experts, only need copy moe experts result to output
            cudaMemcpyAsync(p.output.raw_data(),
                            cutlass_inout_buf_.raw_data(),
                            turbomind::byte_size(cutlass_inout_buf_.dtype()) * tokens * hidden_dim_,
                            cudaMemcpyDeviceToDevice,
                            stream_);
        }
        else 
        {

            auto invoke = [&](auto t) {
                using T = decltype(t);

                // have shared experts, we nedd reduce output(shared experts res) + cutlass_inout_buf_(moe experts res)
                invokeFusedMoeReduce(p.output.data<T>(),
                                     cutlass_inout_buf_.data<T>(),
                                     shared_scales_.data_or((float*)nullptr),
                                     tokens,
                                     hidden_dim_,
                                     p.output_scale,
                                     stream_);
            };

            TM_DISPATCH_DTYPES(cutlass_inout_buf_.dtype(), invoke, half_t, bfloat16_t);
        }
        sync_check_cuda_error();
    }
    else
#endif
    {
        invokeMoeCombine(p.output,
                         temp_,
                         scales_.data(),
                         en2f_.data(),
                         shared_scales_.data_or((float*)nullptr),
                         param_.experts_per_token,
                         p.scale,
                         stream_);
        sync_check_cuda_error();
    }
    // clang-format on

    temp_          = {};
    shared_scales_ = {};
    cutlass_inout_buf_ = {};
}

}  // namespace turbomind