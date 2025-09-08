
#include <utility>

#include "src/turbomind/models/llama/LlamaDenseWeight.h"

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/data_type.h"

#include "src/turbomind/kernels/gemm/cast.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gpt_kernels.h"

#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

void LlamaDenseWeight::emplace(int       input_dim,
                               int       output_dim,
                               DataType  data_type,
                               bool      bias,
                               DataType  weight_type,
                               int       group_size,
                               QuantMode quant_mode,
                               int       expert_num)
{
    this->data_type   = data_type;
    this->weight_type = weight_type;
    this->input_dim   = input_dim;
    this->output_dim  = output_dim;
    this->group_size  = group_size;
    this->quant_mode  = quant_mode;

    bool is_qweight = weight_type == kUint4 || weight_type == kUint8;

    // NOTE(Alan): fp8_static quant only support TN means weight is T
    if ((weight_type == kFloat8_e4m3 && quant_mode.isFP8Static())) {
        weight     = Tensor({output_dim, input_dim}, weight_type, kDEVICE);
        is_qweight = true;
    }
    else {
        weight = Tensor({input_dim, output_dim}, weight_type, kDEVICE);
    }

    register_parameter(is_qweight ? "qweight" : "weight", weight);

    if (bias) {
        this->bias = Tensor{{output_dim}, data_type, kDEVICE};
        register_parameter("bias", this->bias);
    }

    if (weight_type == kFloat8_e4m3) {
        if (quant_mode.isFP8Static()) {
            // NOTE(Alan): 默认在fp8 per tensor static quant的方法
            input_scale = Tensor({1}, DataType::kFloat32, kDEVICE);
            register_parameter("input_scale", input_scale);
            input_scale_inv = Tensor({1}, DataType::kFloat32, kDEVICE);
            register_parameter("input_scale_inv", input_scale_inv);
            weight_scale = Tensor({1*expert_num}, DataType::kFloat32, kDEVICE);
            register_parameter("weight_scale", weight_scale);
            weight_scale_inv = Tensor({1*expert_num}, DataType::kFloat32, kDEVICE);
            register_parameter("weight_scale_inv", weight_scale_inv);
            host_input_scale_inv = Tensor({1}, DataType::kFloat32, kCPU);
            register_parameter("host_input_scale_inv", host_input_scale_inv);
        }
        else {
            // NOTE(Alan): 默认在fp8 per blocks的方法，128x128 block scales，activation dynamic quant
            scales = Tensor{{cdiv(input_dim, 128), cdiv(output_dim, 128)}, kFloat, kDEVICE};
            register_parameter("scales", scales);
        }
    }
    else if (is_qweight) {
        if (quant_mode.isW4A8AWQ()) {
            // TODO(Alan): 进行w4a8 权重加载
        }
        else {
            TM_CHECK(input_dim % group_size == 0) << input_dim << " " << group_size;
            scales = Tensor{{input_dim / group_size, output_dim}, data_type, kDEVICE};
            zeros  = Tensor{{input_dim / group_size, output_dim}, data_type, kDEVICE};
            register_parameter("scales", scales);
            register_parameter("zeros", zeros);
        }
    }
}

static void convert_u4(LlamaDenseWeight& dense, bool is_fused_moe, bool use_simt, cudaStream_t st)
{
    TM_CHECK_EQ(dense.weight_type, data_type_v<uint4_t>);

    using namespace gemm;

    auto [order_b, pack_b, order_v, pack_v] =
        get_weight_and_scales_layout(data_type_v<uint4_t>, is_fused_moe, getSMVersion(), use_simt);

    if (order_b == kColMajor) {
        Buffer trans{dense.input_dim * dense.output_dim, data_type_v<uint4_t>, kDEVICE};
        transpose_u4(
            (uint4_t*)trans.raw_data(), (const uint4_t*)dense.weight.raw_data(), dense.input_dim, dense.output_dim, st);
        cudaMemcpyAsync(
            dense.weight.raw_data(), trans.raw_data(), dense.input_dim * dense.output_dim / 2, cudaMemcpyDefault, st);
    }

    Buffer_<uint16_t> tmp_w{dense.input_dim * dense.output_dim, kDEVICE};
    extend_to_u16(tmp_w.data(), (const uint4_t*)dense.weight.raw_data(), dense.input_dim * dense.output_dim, st);
    sync_check_cuda_error();

    MatrixLayout w_desc{
        data_type_v<half_t>,
        order_b,
        (int)dense.input_dim,   // k
        (int)dense.output_dim,  // n
        order_b == kRowMajor ? (int)dense.output_dim : (int)dense.input_dim,
    };

    MatrixLayout k_desc = w_desc;
    k_desc.type         = data_type_v<uint4_t>;
    k_desc.pack         = pack_b;

    cudaMemsetAsync(dense.weight.raw_data(), 0, dense.input_dim * dense.output_dim / 2, st);

    FT_CHECK(Convert(tmp_w.data(), w_desc, dense.weight.raw_data(), k_desc, st) == 0);
    sync_check_cuda_error();

    const int scale_count = (dense.input_dim / dense.group_size) * dense.output_dim;

    Buffer_<half> tmp_q{scale_count * 2, kDEVICE};
    fuse_scales_and_zeros(tmp_q.data(), dense.scales.data<half>(), dense.zeros.data<half>(), scale_count, st);
    sync_check_cuda_error();

    dense.scales = {};
    dense.zeros  = {};

    dense.scales_zeros = Tensor_<half>{{scale_count, 2}, kDEVICE};

    MatrixLayout s_desc{
        data_type_v<uint32_t>,
        order_v,
        (int)dense.input_dim / dense.group_size,  // k
        (int)dense.output_dim,                    // n
        (int)dense.output_dim,
    };

    MatrixLayout q_desc = s_desc;
    q_desc.pack         = pack_v;

    FT_CHECK(Convert(tmp_q.data(), s_desc, dense.scales_zeros.raw_data(), q_desc, st) == 0);
    sync_check_cuda_error();

    dense.k_desc = k_desc;
    dense.q_desc = q_desc;
}

static void convert_fp(LlamaDenseWeight& dense, bool is_fused_moe, bool use_simt, cudaStream_t st)
{
    using namespace gemm;

    if (!is_fused_moe) {
        return;
    }

    /// TODO: unify data types
    auto data_type = dense.data_type;

    const auto [order_b, pack_b, order_v, pack_v] =
        get_weight_and_scales_layout(data_type, is_fused_moe, getSMVersion(), use_simt);

    const int input_dim  = dense.input_dim;
    const int output_dim = dense.output_dim;

    TM_CHECK(dense.weight.is_contiguous());

    Buffer_<uint16_t> tmp{input_dim * output_dim, kDEVICE};

    if (order_b == kColMajor) {
        invokeTransposeAxis01(tmp.data(), (uint16_t*)dense.weight.raw_data(), input_dim, output_dim, 1, st);
        sync_check_cuda_error();
    }
    else {
        check_cuda_error(
            cudaMemcpyAsync(tmp.data(), dense.weight.raw_data(), dense.weight.byte_size(), cudaMemcpyDefault, st));
    }

    MatrixLayout src{
        data_type,
        order_b,
        input_dim,   // k
        output_dim,  // n
        order_b == kRowMajor ? output_dim : input_dim,
    };

    MatrixLayout dst = src;
    dst.pack         = pack_b;

    if (pack_b) {
        FT_CHECK(Convert(tmp.data(), src, dense.weight.raw_data(), dst, st) == 0);
        sync_check_cuda_error();
    }
    else {
        check_cuda_error(
            cudaMemcpyAsync(dense.weight.raw_data(), tmp.data(), dense.weight.byte_size(), cudaMemcpyDefault, st));
    }

    dense.k_desc = dst;
}

static void convert_f8(LlamaDenseWeight& dense, cudaStream_t stream)
{
    using namespace gemm;

    auto process = [&](Tensor& x, MatrixLayout& d, auto dtype) {
        using T = decltype(dtype);
        Tensor trans{{x.shape(1), x.shape(0)}, x.dtype(), kDEVICE};
        invokeTransposeAxis01((T*)trans.raw_data(), (T*)x.raw_data(), x.shape(0), x.shape(1), 1, stream);
        x = std::move(trans);
        d = MatrixLayout{x.dtype(),  //
                         kColMajor,
                         (int)x.shape(1),
                         (int)x.shape(0),
                         (int)x.stride(0)};
    };

    TM_CHECK_EQ(dense.weight.dtype(), kFloat8_e4m3);
    process(dense.weight, dense.k_desc, uint8_t{});

    TM_CHECK_EQ(dense.scales.dtype(), kFloat);
    process(dense.scales, dense.q_desc, float{});
}

void LlamaDenseWeight::prepare(bool fused_moe, bool use_simt)
{
    if (!weight) {
        return;
    }

    auto stream = core::Context::stream().handle();
    if (weight_type == data_type_v<uint4_t>) {
        convert_u4(*this, fused_moe, use_simt, stream);
    }
    else if (weight_type == data_type_v<fp8_e4m3_t> && quant_mode.isFP8BlockScales()) {
        convert_f8(*this, stream);
    }
    else {
        convert_fp(*this, fused_moe, use_simt, stream);
    }
}

LlamaAttentionWeight::LlamaAttentionWeight(int       hidden_dim,
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
                                           int       group_size,
                                           QuantMode quant_mode)
{
    if (mla.kv_lora_rank == 0) {
        qkv.emplace(hidden_dim,
                    (head_num + 2 * kv_head_num) * head_dim / tp_size,
                    data_type,
                    bias,
                    weight_type,
                    group_size,
                    quant_mode);
        register_module("w_qkv", qkv, tp_rank);
        if (qk_norm) {
            q_a_layernorm  = Tensor{{head_dim}, data_type, kDEVICE};
            kv_a_layernorm = Tensor{{head_dim}, data_type, kDEVICE};
            register_parameter("q_norm", q_a_layernorm);
            register_parameter("k_norm", kv_a_layernorm);
        }
    }
    else {
        const int qk_nope_dim = head_dim - mla.qk_rope_dim;
        if (mla.q_lora_rank) {
            q_a_proj.emplace(hidden_dim, mla.q_lora_rank, data_type, false, weight_type, group_size, quant_mode);
            q_b_proj.emplace(
                mla.q_lora_rank, head_num * head_dim / tp_size, data_type, false, weight_type, group_size, quant_mode);
            q_a_layernorm = Tensor{{q_b_proj.input_dim}, data_type, kDEVICE};
            register_module("q_a_proj", q_a_proj);
            register_module("q_b_proj", q_b_proj, tp_rank);
            register_parameter("q_a_layernorm", q_a_layernorm);
        }
        else {
            q_proj.emplace(hidden_dim, head_num * head_dim / tp_size, data_type, false, weight_type, group_size, quant_mode);
            register_module("q_proj", q_proj, tp_rank);
        }
        kv_a_proj.emplace(
            hidden_dim, mla.kv_lora_rank + mla.qk_rope_dim, data_type, false, weight_type, group_size, quant_mode);
        kv_b_proj.emplace(mla.kv_lora_rank,
                          head_num * (qk_nope_dim + mla.v_head_dim) / tp_size,
                          data_type,
                          false,
                          weight_type,
                          group_size,
                          quant_mode);

        kv_a_layernorm = Tensor{{kv_b_proj.input_dim}, data_type, kDEVICE};
        register_module("kv_a_proj", kv_a_proj);
        register_module("kv_b_proj", kv_b_proj, tp_rank);
        register_parameter("kv_a_layernorm", kv_a_layernorm);
    }
    output.emplace((head_num * head_dim) / tp_size, hidden_dim, data_type, bias, weight_type, group_size, quant_mode);
    register_module("wo", output, tp_rank);

    this->quant_mode = quant_mode;
}

void LlamaAttentionWeight::prepare(bool use_simt)
{
    std::vector weights{
        &qkv,
        &output,
        &q_a_proj,
        &q_a_proj,
        &q_b_proj,
        &kv_a_proj,
        &kv_b_proj,
    };
    for (auto& w : weights) {
        w->prepare(false, use_simt);
    }
}

LlamaFfnWeight::LlamaFfnWeight(int       hidden_dim,
                               int       inter_size,
                               int       tp_size,
                               int       tp_rank,
                               DataType  data_type,
                               DataType  weight_type,
                               int       group_size,
                               bool      fuse_silu_act,
                               QuantMode quant_mode)
{
    TM_CHECK(inter_size % tp_size == 0) << inter_size << " " << tp_size;

    inter_size /= tp_size;

    this->inter_size = inter_size;

    gating.emplace(hidden_dim, inter_size, data_type, false, weight_type, group_size, quant_mode);

    intermediate.emplace(hidden_dim, inter_size, data_type, false, weight_type, group_size, quant_mode);

    // fused_gating_intermediate = {hidden_dim, inter_size * 2, data_type, weight_type, group_size};
    is_fused_silu = fuse_silu_act;

    output.emplace(inter_size, hidden_dim, data_type, false, weight_type, group_size, quant_mode);

    register_module("w1", gating, tp_rank);
    register_module("w3", intermediate, tp_rank);
    register_module("w2", output, tp_rank);

    this->quant_mode = quant_mode;
}

void interleave(LlamaDenseWeight& c, LlamaDenseWeight& a, LlamaDenseWeight& b, DataType data_type, cudaStream_t st)
{
    FT_CHECK(c.input_dim == a.input_dim);
    FT_CHECK(c.input_dim == b.input_dim);
    FT_CHECK(c.output_dim == a.output_dim * 2);
    FT_CHECK(c.output_dim == b.output_dim * 2);
    FT_CHECK(c.group_size == a.group_size);
    FT_CHECK(c.group_size == b.group_size);

    auto invoke = [&](auto t) {
        using T = decltype(t);
        if (a.weight_type == data_type_v<uint4_t>) {
            Buffer_<uint8_t> tmp_a{a.weight.size(), kDEVICE};
            Buffer_<uint8_t> tmp_b{b.weight.size(), kDEVICE};
            Buffer_<uint8_t> tmp_c{c.weight.size(), kDEVICE};

            extend_to_u8(tmp_a.data(), (const uint4_t*)a.weight.raw_data(), a.output_dim * a.input_dim, st);
            extend_to_u8(tmp_b.data(), (const uint4_t*)b.weight.raw_data(), b.output_dim * b.input_dim, st);

            interleave_output_dims(tmp_c.data(), tmp_a.data(), tmp_b.data(), a.output_dim, a.input_dim, st);

            compact_to_u4((uint4_t*)c.weight.raw_data(), tmp_c.data(), c.output_dim * c.input_dim, st);

            interleave_output_dims(c.scales.data<T>(),
                                   a.scales.data<T>(),
                                   b.scales.data<T>(),
                                   a.output_dim,
                                   a.input_dim / a.group_size,
                                   st);
            interleave_output_dims(c.zeros.data<T>(),  //
                                   a.zeros.data<T>(),
                                   b.zeros.data<T>(),
                                   a.output_dim,
                                   a.input_dim / a.group_size,
                                   st);
        }
        else {
            interleave_output_dims(
                c.weight.data<T>(), a.weight.data<T>(), b.weight.data<T>(), a.output_dim, a.input_dim, st);
        }
        // Check at function level
        sync_check_cuda_error();
    };

    TM_DISPATCH_DTYPES(data_type, invoke, half_t, bfloat16_t);
}

void chunk(LlamaDenseWeight& c, LlamaDenseWeight& a, LlamaDenseWeight& b, DataType data_type, cudaStream_t st)
{
    FT_CHECK(c.input_dim == a.input_dim);
    FT_CHECK(c.input_dim == b.input_dim);
    FT_CHECK(c.output_dim == a.output_dim * 2);
    FT_CHECK(c.output_dim == b.output_dim * 2);
    FT_CHECK(c.group_size == a.group_size);
    FT_CHECK(c.group_size == b.group_size);

    auto _chunks = [&](auto c, auto a, auto b, int height, int width) {
        check_cuda_error(
            cudaMemcpy2DAsync((char*)c + 0x000, width * 2, a, width, width, height, cudaMemcpyDefault, st));
        check_cuda_error(
            cudaMemcpy2DAsync((char*)c + width, width * 2, b, width, width, height, cudaMemcpyDefault, st));
    };

    // TODO: remove unused branches
    auto invoke = [&](auto t) {
        using T = decltype(t);
        if (c.weight_type == data_type_v<uint4_t>) {
            _chunks(c.weight.raw_data(), a.weight.raw_data(), b.weight.raw_data(), a.input_dim, 4 * a.output_dim / 8);
            _chunks(c.scales.data<T>(),
                    a.scales.data<T>(),
                    b.scales.data<T>(),
                    a.input_dim / a.group_size,
                    sizeof(T) * a.output_dim);
            _chunks(c.zeros.data<T>(),
                    a.zeros.data<T>(),
                    b.zeros.data<T>(),
                    a.input_dim / a.group_size,
                    sizeof(T) * a.output_dim);
        }
        else {
            _chunks(c.weight.data<T>(), a.weight.data<T>(), b.weight.data<T>(), a.input_dim, sizeof(T) * a.output_dim);
        }
        // Check at function level
        sync_check_cuda_error();
    };

    if (c.weight_type == kFloat8_e4m3) {
        _chunks(c.scales.data<float>(),
                a.scales.data<float>(),
                b.scales.data<float>(),
                cdiv(a.input_dim, a.group_size),
                sizeof(float) * cdiv(a.output_dim, a.group_size));
        _chunks(c.weight.data<fp8_e4m3_t>(),
                a.weight.data<fp8_e4m3_t>(),
                b.weight.data<fp8_e4m3_t>(),
                a.input_dim,
                a.output_dim);
    }
    else {
        TM_DISPATCH_DTYPES(data_type, invoke, half_t, bfloat16_t);
    }
}

void LlamaFfnWeight::prepare(bool fused_moe, bool use_simt)
{
    const auto data_type = gating.data_type;

    auto stream = core::Context().stream().handle();

    if (gating.weight_type == DataType::kFloat8_e4m3 && quant_mode.isFP8Static()) {

        auto& fused_up_and_gate = fused_gating_intermediate;

        fused_up_and_gate.emplace(gating.input_dim,  //
                                  gating.output_dim * 2,
                                  gating.data_type,
                                  false,
                                  gating.weight_type,
                                  gating.group_size,
                                  quant_mode);

        auto& w1w3_weight = fused_gating_intermediate;

        w1w3_weight.input_scale          = gating.input_scale;
        w1w3_weight.input_scale_inv      = gating.input_scale_inv;
        w1w3_weight.host_input_scale_inv = gating.host_input_scale_inv;

        // fused_w1w3 scale
        // # d0_scale = w1_is * w3_ws # w3 intermediate
        // # d1_scale = w1_is * w1_ws # w1 gating
        float host_w3_ws = 0.0, host_w1_ws = 0.0, host_w1_is = 0.0;
        check_cuda_error(
            cudaMemcpyAsync(&host_w3_ws, intermediate.weight_scale.data<float>(), sizeof(float), cudaMemcpyDefault));
        check_cuda_error(
            cudaMemcpyAsync(&host_w1_ws, gating.weight_scale.data<float>(), sizeof(float), cudaMemcpyDefault));
        check_cuda_error(
            cudaMemcpyAsync(&host_w1_is, intermediate.input_scale.data<float>(), sizeof(float), cudaMemcpyDefault));

        float host_d0_scale = host_w1_is * host_w3_ws;
        float host_d1_scale = host_w1_is * host_w1_ws;

        w1w3_weight.host_d0_scale = Tensor({1}, DataType::kFloat32, kCPU);
        w1w3_weight.host_d1_scale = Tensor({1}, DataType::kFloat32, kCPU);

        (*(w1w3_weight.host_d0_scale.data<float>())) = host_d0_scale;
        (*(w1w3_weight.host_d1_scale.data<float>())) = host_d1_scale;

        w1w3_weight.d0_scale = Tensor({1}, DataType::kFloat32, kDEVICE);
        w1w3_weight.d1_scale = Tensor({1}, DataType::kFloat32, kDEVICE);
        check_cuda_error(cudaMemcpyAsync(w1w3_weight.d0_scale.data<float>(),
                                         w1w3_weight.host_d0_scale.data<float>(),
                                         sizeof(float),
                                         cudaMemcpyDefault));
        check_cuda_error(cudaMemcpyAsync(w1w3_weight.d1_scale.data<float>(),
                                         w1w3_weight.host_d1_scale.data<float>(),
                                         sizeof(float),
                                         cudaMemcpyDefault));

        // fused_w1w3 weight
        auto w1w3_weight_ptr = w1w3_weight.weight.data<__nv_fp8_e4m3>();
        auto w1_or_w3_size   = gating.input_dim * gating.output_dim;

        check_cuda_error(cudaMemcpyAsync(w1w3_weight_ptr, 
                                        intermediate.weight.data<__nv_fp8_e4m3>(),
                                        w1_or_w3_size * sizeof(__nv_fp8_e4m3), 
                                        cudaMemcpyDefault,
                                        stream));
        check_cuda_error(cudaMemcpyAsync(w1w3_weight_ptr + w1_or_w3_size, 
                                         gating.weight.data<__nv_fp8_e4m3>(),
                                         w1_or_w3_size * sizeof(__nv_fp8_e4m3), 
                                         cudaMemcpyDefault,
                                         stream));
        return;
    }
    if (fuse_up_and_gate) {
        auto& fused_up_and_gate = fused_gating_intermediate;

        fused_up_and_gate.emplace(gating.input_dim,  //
                                  gating.output_dim * 2,
                                  gating.data_type,
                                  false,
                                  gating.weight_type,
                                  gating.group_size,
                                  quant_mode);
        if (is_fused_silu) {
            interleave(fused_up_and_gate, gating, intermediate, data_type, stream);
        }
        else {
            chunk(fused_up_and_gate, gating, intermediate, data_type, stream);
        }

        fused_gating_intermediate.prepare(fused_moe, use_simt);

        gating       = {};
        intermediate = {};
    }
    else {
        gating.prepare(fused_moe, use_simt);
        intermediate.prepare(fused_moe, use_simt);
    }

    output.prepare(fused_moe, use_simt);
}

MoeFfnWeight::MoeFfnWeight(int             layer_id,
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
                           bool            cutlass_fused_kernel)
{
    if ((int)param.expert_num.size() <= layer_id) {
        return;
    }

    const int expert_num = param.expert_num[layer_id];

    if (expert_num == 0) {
        return;
    }

    gate.emplace(hidden_dim, ep_size * expert_num, data_type, false, data_type, 1, quant_mode);
    register_module("gate", gate);

    method        = param.method;
    fuse_silu_act = fuse_silu_act && method == MoeParam::kFused && weight_type != kFloat8_e4m3;

    experts.reserve(expert_num);
    for (int i = expert_num * ep_rank; i < expert_num * (ep_rank+1); ++i) {
        experts.emplace_back(new LlamaFfnWeight{
            hidden_dim, param.inter_size, tp_size, tp_rank, data_type, weight_type, group_size, fuse_silu_act, quant_mode});
        register_module("experts", *experts.back(), i);
    }

    if (param.shared_gate) {
        shared_gate.emplace(hidden_dim, 1, data_type, false, data_type, 1, quant_mode);
        register_module("shared_gate", shared_gate);
    }

    this->quant_mode = quant_mode;
}


#ifdef FUSED_MOE_FFN_GEMM

void weight_inv(LlamaDenseWeight& weight, cudaStream_t stream)
{
    invokeConvertWeightToInv(weight.weight_scale_inv.data<float>(), weight.weight_scale.data<float>(), 1, stream);
    invokeConvertWeightToInv(weight.input_scale_inv.data<float>(), weight.input_scale.data<float>(), 1, stream);
    cudaMemcpyAsync(weight.host_input_scale_inv.data<float>(), weight.input_scale_inv.data<float>(), sizeof(float), cudaMemcpyDeviceToHost, stream);
}

void fused_experts_weight(LlamaDenseWeight& w1,
                          LlamaDenseWeight& w3,
                          LlamaDenseWeight& w2,
                          LlamaDenseWeight& fused_gating_intermediate,
                          LlamaDenseWeight& out,
                          const int         expert_idx,
                          const int         expert_num)
{
    TM_LOG_TRACE("fused_experts_weight begin");

    // Note(meng): Rearrange the MoE-FFN experts weights
    // fused w1w3
    // TRT-LLM w1/w3 is different from LMDeploy
    const auto st       = core::Context::stream().handle();
    float max_host_w1w3_input_scale_inv = 0.0;
    float max_host_w2_input_scale_inv   = 0.0;

    if (expert_idx == 0) {
        max_host_w1w3_input_scale_inv = 0.0;
        max_host_w2_input_scale_inv   = 0.0;
    }

    if (w1.weight_type == DataType::kFloat8_e4m3 && w3.weight_type == DataType::kFloat8_e4m3 && w2.weight_type == DataType::kFloat8_e4m3) {
        using weight_datatype = __nv_fp8_e4m3;

        // 1. rescale w1_w3 and fused weight
        // find max weight_scale
        float host_w1_scale = 0.0, host_w3_scale = 0.0, host_max_w1w3_wscale = 0.0;
        cudaMemcpyAsync(&host_w1_scale, w1.weight_scale.data<float>(), sizeof(float), cudaMemcpyDefault, st);
        cudaMemcpyAsync(&host_w3_scale, w3.weight_scale.data<float>(), sizeof(float), cudaMemcpyDefault, st);
        host_max_w1w3_wscale = std::max(host_w1_scale, host_w3_scale);

        // rescale weight
        //void invokeRescaleWeight(T* weight, float w_s, int s, int c, cudaStream_t stream)
        invokeRescaleWeight(w1.weight.data<__nv_fp8_e4m3>(),
                            (host_w1_scale / host_max_w1w3_wscale),
                            w1.output_dim,
                            w1.input_dim,
                            st);
        invokeRescaleWeight(w3.weight.data<__nv_fp8_e4m3>(),
                            (host_w3_scale / host_max_w1w3_wscale),
                            w3.output_dim,
                            w3.input_dim,
                            st);

        // fused weight
        const int        expert_w1_size     = w1.input_dim * w1.output_dim;
        const int        expert_w3_size     = w3.input_dim * w3.output_dim;
        const int        expert_offset_w1w3 = expert_idx * (expert_w1_size + expert_w3_size);
        weight_datatype* cur_out_ptr_w3 = fused_gating_intermediate.weight.data<__nv_fp8_e4m3>() + expert_offset_w1w3;
        weight_datatype* cur_out_ptr_w1 = cur_out_ptr_w3 + expert_w3_size;
        check_cuda_error(
            cudaMemcpyAsync(cur_out_ptr_w3, w3.weight.data<__nv_fp8_e4m3>(), sizeof(weight_datatype) * expert_w3_size, cudaMemcpyDeviceToDevice, st));
        check_cuda_error(
            cudaMemcpyAsync(cur_out_ptr_w1, w1.weight.data<__nv_fp8_e4m3>(), sizeof(weight_datatype) * expert_w1_size, cudaMemcpyDeviceToDevice, st));

        // set fused_w1w3 weight scale
        float* w1w3_weight_scale = fused_gating_intermediate.d0_scale.data<float>() + expert_idx;
        check_cuda_error(cudaMemcpyAsync(w1w3_weight_scale, &host_max_w1w3_wscale, sizeof(float), cudaMemcpyHostToDevice, st));

        // Note(meng): check check: all expert have the same host_input_scale_inv
        // printf("expert-id: %d, w1_input_inv: %f, w3_input_inv: %f\n", expert_idx, *(w1.host_input_scale_inv.data<float>()), *(w3.host_input_scale_inv.data<float>()));
        max_host_w1w3_input_scale_inv = std::max(max_host_w1w3_input_scale_inv, *(w1.host_input_scale_inv.data<float>()));

        // 2.fused w2
        const int expert_w2_size   = w2.input_dim * w2.output_dim;
        const int expert_offset_w2 = expert_idx * expert_w2_size;

        weight_datatype* cur_out_ptr_w2 = static_cast<weight_datatype*>(out.weight.data<__nv_fp8_e4m3>()) + expert_offset_w2;
        check_cuda_error(
            cudaMemcpyAsync(cur_out_ptr_w2, w2.weight.data<__nv_fp8_e4m3>(), sizeof(weight_datatype) * expert_w2_size, cudaMemcpyDeviceToDevice, st));

        float* w2_weight_scale = out.d0_scale.data<float>() + expert_idx;
        float cur_host_w2_weight_scale = 1.0;
        check_cuda_error(cudaMemcpyAsync(&cur_host_w2_weight_scale, w2.weight_scale.data<float>(), sizeof(float), cudaMemcpyDeviceToHost, st));
        float cur_host_w2_ab_scale = cur_host_w2_weight_scale * (1.0 / *(w2.host_input_scale_inv.data<float>()));
        check_cuda_error(cudaMemcpyAsync(w2_weight_scale, &cur_host_w2_ab_scale, sizeof(float), cudaMemcpyHostToDevice, st));

        // Note(meng): save moe grouped-gemm2 input_scale_inv in d1_scale (Per-Expert)
        float* w2_input_scale_inv = out.d1_scale.data<float>() + expert_idx;
        check_cuda_error(cudaMemcpyAsync(w2_input_scale_inv, w2.host_input_scale_inv.data<float>(), sizeof(float), cudaMemcpyHostToDevice, st));
        // Note(meng): Under (Per-Expert) quant for moe grouped-gemm2 input_scale_inv. In fact no need cal max_w2_input_scale_inv
        max_host_w2_input_scale_inv = (max_host_w2_input_scale_inv > *(w2.host_input_scale_inv.data<float>())) ?
                                          max_host_w2_input_scale_inv :
                                          *(w2.host_input_scale_inv.data<float>());

        // all expert shared with one input scale
        if (expert_idx == expert_num - 1) {
            // TRT-LLM FP8 Quant input_scale
            float min_host_w1w3_input_scale = 1.0 / max_host_w1w3_input_scale_inv;
            float min_w2_input_scale        = 1.0 / max_host_w2_input_scale_inv;

            // set input_scale
            check_cuda_error(cudaMemcpyAsync(fused_gating_intermediate.input_scale.data<float>(),
                                        &min_host_w1w3_input_scale,
                                        sizeof(float),
                                        cudaMemcpyHostToDevice, st));
            check_cuda_error(cudaMemcpyAsync(out.input_scale.data<float>(), &min_w2_input_scale, sizeof(float), cudaMemcpyHostToDevice, st));

            // rescale d0 scale
            invokeRescaleWeight(
                fused_gating_intermediate.d0_scale.data<float>(), min_host_w1w3_input_scale, expert_num, 1, st);
            //invokeRescaleWeight(static_cast<float*>(out.d0_scale), min_w2_input_scale, expert_num, 1);

            weight_inv(fused_gating_intermediate, st);
            weight_inv(out, st);
        }
    }
    TM_LOG_TRACE("fused_experts_weight end");
}

void MoeFfnWeight::process_fp8_moe_weight()
{
    TM_LOG_TRACE("process_fp8_moe_weight begin");
    // Note(meng): Where to handle the logic of TP and EP ???
    cudaDeviceSynchronize();

    // Note(meng): Only support all experts to have the same size and config!
    for (int idx = 0; idx < experts.size(); idx++) {
        FT_CHECK_WITH_INFO(same_config(*experts[0], *experts[idx]), "Only support all experts to have the same config!");
    }

    const int expert_num = experts.size();

    fused_expert.quant_mode = experts[0]->quant_mode;
    auto& fused_up_and_gate = fused_expert.fused_gating_intermediate;  // datatype: LlamaDenseWeight

    const auto& cur_gating = experts[0]->gating;
    fused_up_and_gate.emplace(cur_gating.input_dim,
                              cur_gating.output_dim * 2 * expert_num,   // fused: (w1 + w3) * expert_num
                              cur_gating.data_type,
                              false,
                              cur_gating.weight_type,
                              cur_gating.group_size,
                              cur_gating.quant_mode,
                              expert_num);

    fused_up_and_gate.d0_scale = Tensor({expert_num}, DataType::kFloat32, kDEVICE);

    // special handling, for moe_fused_weight, has expert_num quant scales. We use d0_scale to save all experts weight
    // scales
    // deviceFree(fused_up_and_gate.weight_scale);
    // deviceFree(fused_up_and_gate.d0_scale);
    // deviceMalloc((float**)&fused_up_and_gate.weight_scale, expert_num);
    // deviceMalloc((float**)&fused_up_and_gate.d0_scale, expert_num);

    auto& fused_output = fused_expert.output;
    fused_output.emplace(cur_gating.input_dim,
                         cur_gating.output_dim * expert_num,  // fused: (w1 + w3) * expert_num
                         cur_gating.data_type,
                         false,
                         cur_gating.weight_type,
                         cur_gating.group_size,
                         cur_gating.quant_mode,
                         expert_num);

    fused_output.d0_scale = Tensor({expert_num}, DataType::kFloat32, kDEVICE);
    fused_output.d1_scale = Tensor({expert_num}, DataType::kFloat32, kDEVICE);
    // mallocWeights(fused_output, false);
    // deviceFree(fused_output.weight_scale);
    // deviceFree(fused_output.d0_scale);
    // // Note(meng): In order to further enhance the accuracy of FP8, support quantization with moe-grouped-gemm2 per-expert input_scale is implemented.
    // // We use d1_scale to save all experts grouped-gemm2 input-scale
    // deviceFree(fused_output.d1_scale);
    // deviceMalloc((float**)&fused_output.weight_scale, expert_num);
    // deviceMalloc((float**)&fused_output.d0_scale, expert_num);
    // deviceMalloc((float**)&fused_output.d1_scale, expert_num);

    // fused all expert weight together and free ptr
    for (int idx = 0; idx < experts.size(); idx++) {
        auto& e = experts[idx];

        cudaDeviceSynchronize();
        fused_experts_weight(e->gating, e->intermediate, e->output, fused_up_and_gate, fused_output, idx, expert_num);
        cudaDeviceSynchronize();

        e->gating       = {};
        e->intermediate = {};
        e->output       = {};
    }

    cudaDeviceSynchronize();
    TM_LOG_TRACE("process_fp8_moe_weight begin");
}

#endif

void MoeFfnWeight::prepare(bool use_simt)
{
    if(experts.size() == 0) return;

    // Note(meng): Moe-Weight prepare in FP8-Static Mode
    if (experts[0]->gating.weight_type == DataType::kFloat8_e4m3 && quant_mode.isFP8Static()) {
        process_fp8_moe_weight();
        return;
    }

    const auto fused_moe = method == MoeParam::kFused;

    for (auto& e : experts) {
        e->prepare(fused_moe, use_simt);
    }
    const int  n_expert = experts.size();
    const auto st       = core::Context::stream().handle();

    auto make_strided_ptr = [&](const auto& ptrs) {
        return std::shared_ptr<void>{gemm::make_strided_ptrs(ptrs, st), [](auto p) { cudaFree(p); }};
    };

    auto make_blocked_ptr = [&](const auto& ptrs) {
        return std::shared_ptr<void>{gemm::make_blocked_ptrs(ptrs, st), [](auto p) { cudaFree(p); }};
    };

    auto process = [&](auto getter) {
        std::vector<std::pair<void*, int>> weight_ptrs;
        std::vector<std::pair<void*, int>> quant_ptrs;

        for (auto& e : experts) {
            auto& m = (*e).*getter;
            weight_ptrs.push_back({m.weight.raw_data(), m.k_desc.ld});
            if (m.scales_zeros) {
                quant_ptrs.emplace_back(m.scales_zeros.raw_data(), m.q_desc.ld);
            }
            else if (m.scales) {
                quant_ptrs.emplace_back(m.scales.raw_data(), m.q_desc.ld);
            }
        }

        LlamaDenseWeight& m = block.*getter;

        {  // Copy properties from exemplar, this assumes all experts has the same shape
            LlamaDenseWeight& e = (*experts.at(0)).*getter;
            m.input_dim         = e.input_dim;
            m.output_dim        = e.output_dim;
            m.group_size        = e.group_size;
            m.data_type         = e.data_type;
            m.weight_type       = e.weight_type;
            m.k_desc            = e.k_desc;
            m.q_desc            = e.q_desc;
        }

        // Dummy tensors to hold the blocked ptrs
        if (m.weight_type == kFloat8_e4m3) {
            TM_CHECK_EQ(quant_ptrs.size(), n_expert);
            m.weight = Tensor{make_blocked_ptr(weight_ptrs), {n_expert}, m.weight_type, kDEVICE};
            m.scales = Tensor{make_blocked_ptr(quant_ptrs), {n_expert}, kFloat, kDEVICE};
        }
        else {
            m.weight = Tensor{make_strided_ptr(weight_ptrs), {n_expert}, m.weight_type, kDEVICE};
            if (!quant_ptrs.empty()) {
                TM_CHECK_EQ(quant_ptrs.size(), n_expert);
                m.scales_zeros = Tensor{make_strided_ptr(quant_ptrs), {n_expert}, m.data_type, kDEVICE};
            }
        }

        m.k_desc.num = m.q_desc.num = experts.size();
    };

    process(&LlamaFfnWeight::fused_gating_intermediate);
    process(&LlamaFfnWeight::output);

    auto& e = *experts.at(0);
    // Copy MLP properties
    block.inter_size    = e.inter_size;
    block.is_fused_silu = e.is_fused_silu;
}

}  // namespace turbomind
