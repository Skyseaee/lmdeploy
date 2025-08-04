

#include <numeric>
#include <optional>

#include <cuda_runtime.h>

#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/moe_ffn_layer.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"
#include "src/turbomind/models/llama/unified_decoder.h"
#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

UnifiedDecoder::UnifiedDecoder(const ModelParam&     model,
                               const EngineParam&    engine,
                               const AttentionParam& attn,
                               const MoeParam&       moe,
                               const LoraParam&      lora,
                               const Context&        ctx):
    layer_num_(model.layer_num),
    hidden_units_(model.hidden_units),
    attn_tp_size_(engine.attn_tp_size),
    attn_dp_size_(engine.attn_dp_size),
    attn_dp_rank_(engine.attn_dp_rank),
    mlp_tp_size_(engine.mlp_tp_size),
    attn_tp_group_(ctx.comm.d_tp_group),
    rmsnorm_eps_(model.norm_eps),
    stream_(ctx.stream),
    d_comm_(ctx.comm.d_comm),
    tune_layer_num_(model.tune_layer_num),
    quant_mode_(model.quant_mode)
{
    attn_layer_ = std::make_unique<UnifiedAttentionLayer>(model, attn, engine, lora, attn_tp_size_, ctx);

    if (std::accumulate(moe.expert_num.begin(), moe.expert_num.end(), 0LL)) {
        moe_ffn_layer_ = std::make_unique<MoeFfnLayer>(model, moe, engine, ctx);
    }

    if (std::accumulate(model.inter_size.begin(), model.inter_size.end(), 0LL)) {
        ffn_layer_ = std::make_unique<LlamaFfnLayer>(model, ctx);
    }

    check_cuda_error(cudaStreamCreateWithFlags(&shared_expert_stream_, cudaStreamNonBlocking));
    check_cuda_error(cudaEventCreateWithFlags(&quant_event_, cudaEventDisableTiming));
    check_cuda_error(cudaEventCreateWithFlags(&shared_expert_event_, cudaEventDisableTiming));

    // TODO(Alan): not support stream parallel when tp > 1
    // TODO(Meng): When tp1 > 1, there is a synchronization issue with the moe_ffn_layer_->Combine, which needs to be fix.
    if (mlp_tp_size_ > 1)
        enable_stream_parallel = false;
}

UnifiedDecoder::~UnifiedDecoder()
{
    check_cuda_error(cudaStreamDestroy(shared_expert_stream_));
    check_cuda_error(cudaEventDestroy(quant_event_));
    check_cuda_error(cudaEventDestroy(shared_expert_event_));

    quant_event_ = shared_expert_event_ = {};
    shared_expert_stream_               = {};
}

void UnifiedDecoder::AllreduceResidualRMSnormAndQauntFP8(Tensor&       hidden_states_fp8,
                                                         Tensor&       residual,
                                                         Tensor&       moe_fp8_buf,
                                                         Tensor&       moe_fp16_buf,
                                                         Tensor&       hidden_states,
                                                         const Tensor& bias,
                                                         const Tensor& weight,
                                                         const float   shared_expert_scale,
                                                         const float   moe_expert_scale,
                                                         int           token_num,
                                                         int           t0,
                                                         int           t1,
                                                         const int*    local_token_nums)
{
    const auto dtype = hidden_states.dtype();

    invokeResidualBiasRMSNormAndQuantFP8(hidden_states_fp8.raw_data(),
                                         residual.raw_data(),
                                         moe_fp8_buf.data_or((void*)nullptr),
                                         moe_fp16_buf.data_or((void*)nullptr),
                                         hidden_states.raw_data(),
                                         weight.raw_data(),
                                         bias.data_or((void*)nullptr),
                                         shared_expert_scale,
                                         moe_expert_scale,
                                         dtype,
                                         hidden_units_,
                                         token_num,
                                         rmsnorm_eps_,
                                         stream_);
    sync_check_cuda_error();
}

void UnifiedDecoder::AllreduceResidualRMSnorm(Tensor&               hidden_states,
                                              Tensor&               residual,
                                              const Tensor&         bias,
                                              const Tensor&         weight,
                                              int                   token_num,
                                              int                   group0,
                                              int                   group1,
                                              const int*            local_token_nums,
                                              std::optional<Tensor> hidden_states_fp8,
                                              std::optional<Tensor> moe_fp8_buf,
                                              std::optional<Tensor> moe_fp16_buf,
                                              float                 shared_expert_scale,
                                              float                 moe_expert_scale)
{
    const auto dtype = hidden_states.dtype();
    if (0) {}
    else if (group0 || group1) {
        d_comm_->AllreduceResidualBiasRMSnormEx(hidden_states.raw_data(),
                                                residual.raw_data(),
                                                bias.data_or((void*)nullptr),
                                                weight.raw_data(),
                                                rmsnorm_eps_,
                                                hidden_units_,
                                                dtype,
                                                group0,
                                                group1,
                                                local_token_nums,
                                                stream_);
        sync_check_cuda_error();
    }
    else if (d_comm_) {
        d_comm_->AllreduceResidualBiasRMSnorm(hidden_states.raw_data(),
                                              residual.raw_data(),
                                              bias.data_or((void*)nullptr),
                                              weight.raw_data(),
                                              rmsnorm_eps_,
                                              hidden_units_,
                                              token_num,
                                              dtype,
                                              0,
                                              stream_,
                                              hidden_states_fp8 ? hidden_states_fp8->data_or((void*)nullptr) : nullptr,
                                              moe_fp8_buf ? moe_fp8_buf->data_or((void*)nullptr) : nullptr,
                                              moe_fp16_buf ? moe_fp16_buf->data_or((void*)nullptr) : nullptr,
                                              shared_expert_scale,
                                              moe_expert_scale);
        sync_check_cuda_error();
    }
    else {
        if (hidden_states_fp8)
            invokeResidualBiasRMSNormAndQuantFP8(hidden_states_fp8->raw_data(),
                                                 residual.raw_data(),
                                                 moe_fp8_buf ? moe_fp8_buf->data_or((void*)nullptr) : nullptr,
                                                 moe_fp16_buf ? moe_fp16_buf->data_or((void*)nullptr) : nullptr,
                                                 hidden_states.raw_data(),
                                                 weight.raw_data(),
                                                 bias.data_or((void*)nullptr),
                                                 shared_expert_scale,
                                                 moe_expert_scale,
                                                 dtype,
                                                 hidden_units_,
                                                 token_num,
                                                 rmsnorm_eps_,
                                                 stream_);
        else
            invokeResidualBiasRMSNorm(hidden_states.raw_data(),
                                      residual.raw_data(),
                                      weight.raw_data(),
                                      bias.data_or((void*)nullptr),
                                      dtype,
                                      hidden_units_,
                                      token_num,
                                      rmsnorm_eps_,
                                      stream_);
        sync_check_cuda_error();
    }
}

void UnifiedDecoder::Forward(TensorMap& args, const std::vector<WeightType*>& weights)
{
    /**
     * input tensors:
     *   \param decoder_input [token_num, hidden_units], float
     *   \param output_norm_weight [hidden_dims], float
     *   \param cu_block_counts [batch_size+1], int
     *   \param finished [batch_size], bool
     *   \param rope_theta [batch_size], float
     *   \param h_q_len [batch_size], int on cpu
     *   \param h_k_len [batch_size], int on cpu
     *   \param pf_batch_size [1], int on cpu
     *   \param dc_batch_size [1], int on cpu
     *
     * output tensors:
     *   \param decoder_output [num_token, hidden_units],
     *   \param last_token_hidden_units [batch_size, hidden_units]
     *   \param block_ptrs [total_block_counts], void*
     */
	
    const int decode_num = *args.at("decode_num").data<int>();
    const int prefil_num = *args.at("prefil_num").data<int>();
    const int batch_size = prefil_num + decode_num;

    constexpr auto device = kDEVICE;

    Tensor_<int> local_token_nums = args.at("local_token_nums");

    Tensor local_residual       = args.at("decoder_input");
    Tensor global_hidden_states = args.at("decoder_output");

    Tensor local_moe_fp8_buf       = args.at("moe_fp8_buf");
    Tensor local_moe_fp16_buf      = args.at("moe_fp16_buf");
    Tensor local_moe_gate_fp32_buf = args.at("moe_gate_fp32_buf");

    Tensor local_hidden_states = global_hidden_states;

    const auto global_token_num = global_hidden_states.shape(0);
    const auto local_token_num  = local_residual.shape(0);

    if (attn_dp_size_ > 1) {  // Offset hidden states buffer for mixed DP
        TM_CHECK_EQ(local_token_nums.size(), attn_dp_size_);
        std::vector cumul_token_nums(attn_dp_size_ + 1, 0);
        std::inclusive_scan(
            local_token_nums.data(), local_token_nums.data() + attn_dp_size_, cumul_token_nums.begin() + 1);
        const int offset    = cumul_token_nums[attn_dp_rank_];
        local_hidden_states = global_hidden_states.slice({offset, 0}, {local_token_num, -1});
    }

    attn_layer_->Initialize(args);

    TM_DEBUG_TENSOR(local_residual, "res", 1);
    TM_DEBUG_TENSOR(weights.at(0)->self_attn_norm, "norm_weight", 2);

#ifdef ENABLE_FP8
    if (weights.at(0)->isPerTensorStaticFP8Weight()) {
        // TODO(Alan): 当前算子实现与上个版本存在差异，后续需要考虑性能优化
        invokeRMSNormAndQuant(local_hidden_states,
                              local_residual,
                              weights.at(0)->self_attn_norm,
                              rmsnorm_eps_,
                              *(weights.at(0)->self_attn_weights->qkv.host_input_scale_inv.data<float>()),
                              stream_);
        sync_check_cuda_error();

        TM_DEBUG_TENSOR(local_hidden_states, Concat("norm0", 0), 2);
    }
    else
#endif
    {
        invokeRMSNorm(local_hidden_states, local_residual, weights.at(0)->self_attn_norm, rmsnorm_eps_, stream_);
        sync_check_cuda_error();

        TM_DEBUG_TENSOR(local_hidden_states, Concat("norm0", 0), 2);
    }

    for (int layer = 0; layer < layer_num_; ++layer) {
        /// TODO: do not skip the layers when they are heterogeneous
        if (isTuning() && layer >= tune_layer_num_) {
            continue;
        }

        bool need_return_fp16_res = false;

        /////////////////////////////////////////////
        /// self-attention
        attn_layer_->Forward({local_hidden_states,  //
                              local_hidden_states,
                              weights.at(layer)->self_attn_weights.get(),
                              layer,
                              local_moe_fp8_buf});

        TM_DEBUG_TENSOR(local_hidden_states, Concat("attn_block", layer), 2);

        if (quant_mode_.isFP8Static()) {

            float shared_expert_scale = 1.0;
            if(weights.at(layer)->ffn_weights)
                shared_expert_scale = *(weights.at(layer)->ffn_weights->gating.host_input_scale_inv.data<float>());

            need_return_fp16_res = weights.at(layer)->moe_weights==nullptr ? false : true;

            float moe_scale =
                (need_return_fp16_res) ?
                    *(weights.at(layer)
                          ->moe_weights->fused_expert.fused_gating_intermediate.host_input_scale_inv.data<float>()) :
                    1.0;
            // AllreduceResidualRMSnormAndQauntFP8(global_hidden_states,
            //                                     local_residual,
            //                                     local_moe_fp8_buf,
            //                                     local_moe_fp16_buf,
            //                                     global_hidden_states,
            //                                     weights.at(layer)->self_attn_weights->output.bias,
            //                                     weights.at(layer)->ffn_norm,
            //                                     shared_expert_scale,
            //                                     moe_scale,
            //                                     local_token_num,
            //                                     attn_tp_group_,
            //                                     0,
            //                                     local_token_nums.data());

            AllreduceResidualRMSnorm(global_hidden_states,
                                     local_residual,
                                     weights.at(layer)->self_attn_weights->output.bias,
                                     weights.at(layer)->ffn_norm,
                                     local_token_num,
                                     attn_tp_group_,
                                     0,
                                     local_token_nums.data(),
                                     global_hidden_states,
                                     local_moe_fp8_buf,
                                     local_moe_fp16_buf,
                                     shared_expert_scale,
                                     moe_scale);
            
            TM_DEBUG_TENSOR(local_moe_fp8_buf, Concat("norm1", layer), 2);
            TM_DEBUG_TENSOR(local_moe_fp16_buf, Concat("norm1", layer), 2);
        }
        else {
            AllreduceResidualRMSnorm(global_hidden_states,
                                     local_residual,
                                     weights.at(layer)->self_attn_weights->output.bias,
                                     weights.at(layer)->ffn_norm,
                                     local_token_num,
                                     attn_tp_group_,
                                     0,
                                     local_token_nums.data());
        }

        TM_DEBUG_TENSOR(local_residual, Concat("residual0", layer), 2);
        TM_DEBUG_TENSOR(local_hidden_states, Concat("norm1", layer), 2);

        ////////////////////////////////////////////
        /// feed-forward network
        cudaStream_t shared_expert_stream = stream_;

        // Note(meng): when we have shared expert and moe experts at the same time in FP8, we use stream parallel
        bool use_shared_stream = enable_stream_parallel && quant_mode_.isFP8Static() && weights.at(layer)->ffn_weights && weights.at(layer)->moe_weights /*&& token_num <= 512(except prefill)*/;
        if(use_shared_stream)
        {
            shared_expert_stream = shared_expert_stream_;
            check_cuda_error(cudaEventRecord(quant_event_, stream_));
            check_cuda_error(cudaStreamWaitEvent(shared_expert_stream_, quant_event_));
        }

        if (weights.at(layer)->ffn_weights && use_shared_stream) {
            ffn_layer_->forward(
                {global_hidden_states, global_hidden_states, weights.at(layer)->ffn_weights.get(), (int)layer, shared_expert_stream});
        }

        
        std::optional<MoeFfnLayer::ForwardParam> moe_fwd_param;

        if (weights.at(layer)->moe_weights) {
            if(quant_mode_.isFP8Static()){
                moe_fwd_param = MoeFfnLayer::ForwardParam{local_moe_fp8_buf,        // input (fp8, already quant)
                                                          global_hidden_states,     // output
                                                          local_moe_fp16_buf,       // input (fp16, for gate)
                                                          local_moe_gate_fp32_buf,  // moe gate result buf
                                                          weights.at(layer)->moe_weights.get(),
                                                          ffn_layer_ ? 1.f : 0.f,
                                                          layer,
                                                          prefil_num,
                                                          1.0,  // output_scale (by default)
                                                          use_shared_stream,
                                                          shared_expert_event_,
                                                          shared_expert_stream_};
            }
            else{
                moe_fwd_param = MoeFfnLayer::ForwardParam{global_hidden_states,     // input (fp16)
                                                          global_hidden_states,     // output
                                                          global_hidden_states,     // input (fp16, for gate)
                                                          local_moe_gate_fp32_buf,  // moe gate result buf
                                                          weights.at(layer)->moe_weights.get(),
                                                          ffn_layer_ ? 1.f : 0.f,
                                                          layer,
                                                          prefil_num};
            }
            moe_ffn_layer_->Forward(*moe_fwd_param);
        }

        if (weights.at(layer)->ffn_weights && !use_shared_stream) {
            ffn_layer_->forward(
                {global_hidden_states, global_hidden_states, weights.at(layer)->ffn_weights.get(), (int)layer});
        }

        // Note(meng): when we use stream parallel, we fused reduce kernel into moe group gemm epilogue, no need moe_ffn_layer_->reduce
        if (moe_fwd_param && !use_shared_stream) {
            moe_ffn_layer_->Combine(*moe_fwd_param);
        }

        TM_DEBUG_TENSOR(global_hidden_states, Concat("ffn_block", layer), 2);

        const bool last = layer == layer_num_ - 1;

        auto& scale_weight = !last ? weights.at(layer + 1)->self_attn_norm : args.at("output_norm_weight");

        if (layer < layer_num_ - 1 && quant_mode_.isFP8Static()) {
            AllreduceResidualRMSnorm(global_hidden_states,
                                     local_residual,
                                     {},
                                     scale_weight,
                                     local_token_num,
                                     0,
                                     attn_tp_group_,
                                     local_token_nums.data(),
                                     global_hidden_states,
                                     std::nullopt,
                                     std::nullopt,
                                     *(weights.at(layer + 1)->self_attn_weights->qkv.host_input_scale_inv.data<float>()));

            // AllreduceResidualRMSnormAndQauntFP8(
            //     global_hidden_states,
            //     local_residual,
            //     empty_tensor,
            //     empty_tensor,
            //     global_hidden_states,
            //     {},
            //     scale_weight,
            //     *(weights.at(layer + 1)->self_attn_weights->qkv.host_input_scale_inv.data<float>()),
            //     1.0,
            //     local_token_num,
            //     0,
            //     attn_tp_group_,
            //     local_token_nums.data());
        }
        else {
            AllreduceResidualRMSnorm(global_hidden_states,
                                     local_residual,
                                     {},
                                     scale_weight,
                                     local_token_num,
                                     0,
                                     attn_tp_group_,
                                     local_token_nums.data());
        }

        sync_check_cuda_error();

        TM_DEBUG_TENSOR(local_residual, Concat("residual1", layer), 2);
        TM_DEBUG_TENSOR(local_hidden_states, Concat("norm0", layer + 1), 2);
    }

    /// TODO
    using T = uint16_t;

    auto last_token_hidden_units = (T*)args.at("last_token_hidden_units").raw_data();

    if (decode_num) {
        check_cuda_error(cudaMemcpyAsync(last_token_hidden_units,
                                         (T*)local_hidden_states.raw_data(),
                                         sizeof(T) * decode_num * hidden_units_,
                                         cudaMemcpyDefault,
                                         stream_));
        // TM_DEBUG_RAW(last_token_hidden_units, decode_num * hidden_units_, "dc_out", 2);
    }

    if (prefil_num) {
        invokeGetFeatureOfLastToken(last_token_hidden_units + decode_num * hidden_units_,  //
                                    (T*)local_hidden_states.raw_data(),
                                    attn_layer_->d_cu_q_len() + decode_num,
                                    hidden_units_,
                                    prefil_num,
                                    stream_);
        sync_check_cuda_error();
        // TM_DEBUG_RAW(last_token_hidden_units + decode_num * hidden_units_, prefil_num * hidden_units_, "pf_out", 2);
    }

    Buffer out(
        (void*)last_token_hidden_units, (decode_num + prefil_num) * hidden_units_, local_residual.dtype(), kDEVICE);

    TM_DEBUG_TENSOR(out, "out", 1);
    attn_layer_->Finalize();
}

}  // namespace turbomind
