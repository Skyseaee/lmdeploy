// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/cuda_data_type.h"
#include "src/turbomind/core/data_type.h"

#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/moe_utils_v2.h"
#include "src/turbomind/kernels/gemm/types.h"

#include "src/turbomind/kernels/quantization.h"

#include "src/turbomind/models/llama/LlamaLinear.h"

#include "src/turbomind/utils/cuda_utils.h"

#include "src/turbomind/kernels/cutlass_w8a8/scaled_mm_entry.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cublasFP8MMWrapper.h"
#include "src/turbomind/utils/cublasMMWrapper.h"
#include "src/turbomind/utils/memory_utils.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <fstream>

#ifdef FUSED_GATED_GEMM
#include "src/turbomind/kernels/fused_gated_gemm/fused_gated_gemm.h"
#include "src/turbomind/kernels/gemm_profiler/gemmPluginProfiler.h"
#include "src/turbomind/kernels/gemm_profiler/gemmSwigluProfiler.h"
#endif

#ifdef FUSED_MOE_FFN_GEMM
#include "src/turbomind/kernels/cutlass_kernels/include/moe_kernels.h"
#include "src/turbomind/kernels/cutlass_kernels/moe_gemm/workspace.h"
#include "src/turbomind/kernels/gemm_profiler/gemmMoEProfiler.h"
#include "src/turbomind/kernels/moe_load_balance_kernels.h"
#endif

namespace turbomind {

// workspace for cublas/cutlass gemm : 64MB
#define CUTLASS_WORKSPACE_SIZE CUBLAS_WORKSPACE_SIZE

namespace tlkc = tensorrt_llm::kernels::cutlass_kernels;

#ifdef FUSED_GATED_GEMM
using GemmSwigluProfilerPtr = std::shared_ptr<tensorrt_llm::plugins::GemmSwigluPluginProfiler>;
using GemmSwigluRunnerPtr   = std::shared_ptr<tlkc::CutlassFusedGatedGemmRunnerInterface>;
using GemmPluginProfilerManager =
    tensorrt_llm::plugins::GemmPluginProfilerManager<tensorrt_llm::plugins::GemmSwigluPluginProfiler>;
#endif

#ifdef FUSED_MOE_FFN_GEMM
using MoEGemmSwigluRunnerPtr          = std::shared_ptr<tlkc::CutlassMoeFCRunnerInterface>;
using MOEParallelismConfig            = tlkc::MOEParallelismConfig;
using MixtureOfExpertsGemmProfilerPtr = std::shared_ptr<tensorrt_llm::plugins::MixtureOfExpertsGemmProfiler>;
using MOEGemmPluginProfilerManager =
    tensorrt_llm::plugins::GemmPluginProfilerManager<tensorrt_llm::plugins::MixtureOfExpertsGemmProfiler>;
#endif

struct LlamaLinear::Impl {

    explicit Impl(cudaStream_t stream, const ModelParam& model, const EngineParam& engine, const MoeParam& moe):
        stream_(stream), model_param_(model), engine_param_(engine), moe_param_(moe)
    {
        workspace_ = {};

        workspace_.barriers_size   = gemm::Gemm::kBarriersSize;
        workspace_.partials_size   = gemm::Gemm::kPartialsSize;
        workspace_.tensormaps_size = 4096 * 128;  // maximum 4096 tensor maps
        workspace_.cublas_size     = CUBLAS_WORKSPACE_SIZE;
        workspace_.cutlass_size    = CUTLASS_WORKSPACE_SIZE;

        check_cuda_error(cudaMallocAsync(&workspace_.barriers, workspace_.barriers_size, stream_));
        check_cuda_error(cudaMallocAsync(&workspace_.partials, workspace_.partials_size, stream_));
        check_cuda_error(cudaMallocAsync(&workspace_.tensormaps, workspace_.tensormaps_size, stream_));
        check_cuda_error(cudaMallocAsync(&workspace_.cublas, workspace_.cublas_size, stream_));
        check_cuda_error(cudaMemsetAsync(workspace_.barriers, 0, workspace_.barriers_size, stream_));
        check_cuda_error(cudaMalloc(&workspace_.flags, sizeof(int)));

        check_cuda_error(cublasCreate(&cublas_));
        check_cuda_error(cublasSetStream(cublas_, stream_));
        check_cuda_error(cublasSetWorkspace(cublas_, workspace_.partials, workspace_.partials_size));

        if (0) {
            check_cuda_error(cublasSetMathMode(cublas_, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
        }

        check_cuda_error(cublasLtCreate(&cublasLt_));
        cublas_algo_map_ = std::make_unique<cublasAlgoMap>("");
        cublas_mutex_    = std::make_unique<std::mutex>();
        cublas_wrapper_  = std::move(std::make_unique<cublasFP8MMWrapper>(cublas_,
                                                                         cublasLt_,
                                                                         stream_,
                                                                         cublas_algo_map_.get(),
                                                                         cublas_mutex_.get(),
                                                                         workspace_.cublas,
                                                                         workspace_.cublas));
        cublas_wrapper_->setFP8GemmConfig();

        if (model.quant_mode.isFP8Static()) {
            tunningFusedGatedGemm();
            tuningFusedMoEGemm(model, engine, moe);
        }
    }

    ~Impl()
    {
        cublasDestroy(cublas_);
        cudaFreeAsync(workspace_.barriers, stream_);
        cudaFreeAsync(workspace_.partials, stream_);
        cudaFreeAsync(workspace_.tensormaps, stream_);
        cudaFreeAsync(workspace_.flags, stream_);

        cublasLtDestroy(cublasLt_);
        cudaFreeAsync(workspace_.cutlass, stream_);
        cudaFreeAsync(workspace_.cublas, stream_);

        workspace_ = {};
    }

    void tunningFusedGatedGemm();

    void tuningFusedMoEGemm(const ModelParam& model, const EngineParam& engine, const MoeParam& moe);

    void onellmFP8Dense(Tensor& output, const Tensor& input, const LlamaDenseWeight& weight, Type type, void* param, cudaStream_t cur_stream = nullptr);

    void onellmFP8MoE(Tensor&               output,
                      Tensor&               input,
                      Tensor&               logits,
                      //Tensor&               inter_buf_fp8,
                      Tensor&               cutlass_inout_buf,
                      const LlamaFfnWeight& weights,
                      int                   tokens,
                      int                   expert_num,
                      bool                  use_shared_stream,
                      cudaEvent_t           shared_expert_event,
                      cudaStream_t          shared_expert_stream);

    void forward(Tensor& output, const Tensor& input, const LlamaDenseWeight& dense, Type type, void* param, cudaStream_t cur_stream)
    {
        // std::cout << dense.weight << std::endl;
        switch (dense.weight_type) {
            case kFloat16:
            case kFloat32:
            case kBfloat16:
                return forwardFp(output, input, dense.weight);
            case kUint4:
                return forwardInt4(output, input, dense, type);
            case kFloat8_e4m3:
                if (dense.quant_mode.isFP8Static()) {
                    return onellmFP8Dense(output, input, dense, type, param, cur_stream);
                }
                else {
                    return forwardFp8(output, input, dense, type, nullptr, nullptr);
                }
            default:
                TM_CHECK(0) << "not implemented for weight type: " << dense.weight_type;
        }
    }

    void forwardFp(Ref<Tensor> output_, const Tensor& input, const Tensor& weight)
    {
        auto& output = output_.get();
        TM_CHECK_EQ(weight.ndim(), 2);
        TM_CHECK_EQ(input.ndim(), 2);
        TM_CHECK_EQ(output.ndim(), 2);

        int m, n, k;
        std::tie(k, m) = weight.shapes(0, 1);
        n              = input.shape(0);

        TM_CHECK_EQ(input.shape(1), k);
        TM_CHECK_EQ(output.shape(0), n);
        TM_CHECK_EQ(output.shape(1), m);

        // [k, m]
        cublasOperation_t transa = weight.stride(1) == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;
        // [n, k]
        cublasOperation_t transb = input.stride(1) == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;

        const float alpha = 1.f;
        const float beta  = 0.f;

        check_cuda_error(cublasGemmEx(cublas_,
                                      transa,
                                      transb,
                                      m,
                                      n,
                                      k,
                                      &alpha,
                                      weight.raw_data(),
                                      to_cuda_dtype(weight.dtype()),
                                      weight.stride(0) * weight.stride(1),  // one of these is 1
                                      input.raw_data(),
                                      to_cuda_dtype(input.dtype()),
                                      input.stride(0) * input.stride(1),  // one of these is 1
                                      &beta,
                                      output.raw_data(),
                                      to_cuda_dtype(output.dtype()),
                                      output.stride(0) * output.stride(1),  // one of these is 1
                                      CUDA_R_32F,
                                      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // NOTE(Alan): for Block-Wise and Dynamic Activation FP8 Quant Support
    void forwardFp8(Tensor&                 output,
                    const Tensor&           input,
                    const LlamaDenseWeight& dense,
                    Type                    type,
                    const int*              offsets,
                    const int*              indices)
    {
        TM_CHECK_EQ(output.ndim(), 2);  // A [m, k]
        TM_CHECK_EQ(input.ndim(), 2);   // C [m, n]

        TM_CHECK_EQ(input.stride(1), 1) << "input must be row-major";
        TM_CHECK_EQ(output.stride(1), 1) << "output must be row-major";

        // TM_CHECK_EQ(output.shape(0), input.shape(0));
        TM_CHECK_EQ(input.shape(1), dense.input_dim);
        // TM_CHECK_EQ(output.shape(1), dense.output_dim);

        using namespace gemm;

        TM_CHECK(type == kGemm);

        Operation operation{dispatch_policy_,  //
                            Epilogue::kNone,
                            {QuantType::kDefault, 128},
                            {QuantType::kDefault, 128},
                            0};

        Tensor quant;
        Tensor scale;
        QuantizeSymm(quant, scale, input, stream_);
        sync_check_cuda_error();

        const int group_num = dense.k_desc.num;

        if (indices) {
            const auto [m, k] = input.shapes(0, 1);
            const int e       = output.shape(0) / m;

            Tensor a_e = {{m * e, k}, quant.dtype(), quant.device()};
            invokeMoeDispatch(a_e, quant, indices, e, stream_);
            sync_check_cuda_error();

            Tensor u_e;
            invokeMoeDispatchScales(u_e, scale, indices, e, stream_);
            sync_check_cuda_error();

            quant = a_e;
            scale = u_e;
        }

        MatrixLayout a_desc{
            quant.dtype(),
            kRowMajor,
            (int)quant.shape(0),
            dense.input_dim,
            (int)quant.stride(0),
        };

        MatrixLayout u_desc{scale.dtype(),  //
                            kColMajor,
                            (int)scale.shape(1),
                            (int)scale.shape(0),
                            (int)scale.stride(0)};

        MatrixLayout c_desc{
            output.dtype(),  //
            kRowMajor,
            (int)output.shape(0),
            dense.output_dim,
            (int)output.stride(0),
        };

        auto b_desc = dense.k_desc;
        auto v_desc = dense.q_desc;

        if (group_num > 1) {
            // clang-format off
            a_desc.offsets = u_desc.offsets = c_desc.offsets = const_cast<int*>(offsets);
            a_desc.num     = u_desc.num     = c_desc.num     = group_num;
            // clang-format on

            // This is needed to be recognized as blocked striding mode
            b_desc.offsets = v_desc.offsets = (int*)1;

            // std::cout << "A: " << a_desc << "\n";
            // std::cout << "U: " << u_desc << "\n";
            // std::cout << "B: " << dense.k_desc << "\n";
            // std::cout << "V: " << dense.q_desc << "\n";
            // std::cout << "C: " << c_desc << "\n";
        }

        auto ec = gemm_.Run(operation,
                            1.f,
                            quant.raw_data(),
                            a_desc,
                            scale.raw_data(),
                            u_desc,
                            dense.weight.raw_data(),
                            b_desc,
                            dense.scales.raw_data(),
                            v_desc,
                            type == kFusedAdd ? 1.0f : 0.0f,
                            output.raw_data(),
                            c_desc,
                            output.raw_data(),
                            c_desc,
                            workspace_,
                            stream_);
        sync_check_cuda_error();

        // if (group_num > 1) {
        //     TM_CHECK(0);
        // }

        if (ec) {
            TM_LOG_ERROR("%s: %d", __PRETTY_FUNCTION__, ec);
        }
    }

    void forwardInt4(Tensor& output, const Tensor& input, const LlamaDenseWeight& dense, Type type)
    {
        TM_CHECK_EQ(output.ndim(), 2);  // A [m, k]
        TM_CHECK_EQ(input.ndim(), 2);   // C [m, n]

        TM_CHECK_EQ(input.stride(1), 1) << "input must be row-major";
        TM_CHECK_EQ(output.stride(1), 1) << "output must be row-major";

        TM_CHECK_EQ(output.shape(0), input.shape(0));
        TM_CHECK_EQ(input.shape(1), dense.input_dim);
        // TM_CHECK_EQ(output.shape(1), dense.output_dim);

        using namespace gemm;

        const Operation operation{dispatch_policy_,
                                  type == kFusedSiluFfn ? Epilogue::kGatedSilu : Epilogue::kNone,
                                  {QuantType::kDefault},
                                  {QuantType::kDefault, dense.group_size},
                                  0,
                                  {},
                                  nullptr};

        const MatrixLayout a_desc{
            input.dtype(),
            kRowMajor,
            (int)input.shape(0),
            dense.input_dim,
            (int)input.stride(0),
        };

        const MatrixLayout c_desc{
            output.dtype(),  //
            kRowMajor,
            (int)output.shape(0),
            dense.output_dim,
            (int)output.stride(0),
            // type == kFusedSiluFfn ? (int)weight.output_dim / 2 : (int)weight.output_dim,
        };

        auto ec = gemm_.Run(operation,
                            1.f,
                            input.raw_data(),
                            a_desc,
                            nullptr,
                            {},
                            dense.weight.raw_data(),
                            dense.k_desc,
                            dense.scales_zeros.raw_data(),
                            dense.q_desc,
                            type == kFusedAdd ? 1.0f : 0.0f,
                            output.raw_data(),
                            c_desc,
                            output.raw_data(),
                            c_desc,
                            workspace_,
                            stream_);

        if (ec) {
            TM_LOG_ERROR("%s: %d", __PRETTY_FUNCTION__, ec);
        }
    }

    void forward_moe(Tensor&                 output,
                     const Tensor&           input,
                     const int*              indexes,
                     const int*              offsets,
                     const LlamaDenseWeight& dense,
                     Type                    type,
                     gemm::Context*          context)
    {
        using namespace gemm;

        QuantDesc quant_b{};
        if (dense.k_desc.type == kUint4) {
            quant_b.type       = QuantType::kDefault;
            quant_b.group_size = dense.group_size;
        }

        const Operation operation{dispatch_policy_,
                                  type == kFusedSiluFfn ? Epilogue::kGatedSilu : Epilogue::kNone,
                                  {QuantType::kDefault},
                                  quant_b,
                                  0,
                                  context,
                                  nullptr};

        MatrixLayout a_desc{
            input.dtype(),
            kRowMajor,
            (int)output.shape(0),  // batch size
            dense.input_dim,       // k
            (int)input.stride(0),
        };

        a_desc.offsets = (int*)offsets;
        a_desc.idxs    = (int*)indexes;

        // std::cout << "m" << batch_size << "n" << weight.output_dims << "k" << weight.input_dims << " "
        //           << input_data.pitch << "\n";

        MatrixLayout c_desc{
            output.dtype(),  //
            kRowMajor,
            (int)output.shape(0),  // batch size
            dense.output_dim,
            (int)output.stride(0),
            // type == kFusedSiluFfn ? (int)weight.output_dims / 2 : (int)weight.output_dims,
        };

        c_desc.offsets = (int*)offsets;

        a_desc.num = c_desc.num = dense.k_desc.num;

        auto k_desc = dense.k_desc;
        auto q_desc = dense.q_desc;
        // pre-90 grouped gemm need `ld == 0` to resolve with strided_ptr
        k_desc.ld = q_desc.ld = 0;

        auto ec = gemm_.Run(operation,
                            1.f,
                            input.raw_data(),
                            a_desc,
                            nullptr,
                            {},
                            dense.weight.raw_data(),
                            k_desc,
                            dense.scales_zeros.data_or((void*)nullptr),
                            q_desc,
                            type == kFusedAdd ? 1.0f : 0.0f,
                            output.raw_data(),
                            c_desc,
                            output.raw_data(),
                            c_desc,
                            workspace_,
                            stream_);

        if (ec) {
            TM_LOG_ERROR("%s: %d", __PRETTY_FUNCTION__, ec);
        }
    }

public:
    cublasHandle_t       cublas_;
    cublasLtHandle_t     cublasLt_;
    gemm::Gemm           gemm_;
    gemm::DispatchPolicy dispatch_policy_{gemm::DispatchPolicy::kDefault};
    cudaStream_t         stream_{};

    gemm::Workspace workspace_;

    std::unique_ptr<cublasAlgoMap>      cublas_algo_map_;
    std::unique_ptr<std::mutex>         cublas_mutex_;
    std::unique_ptr<cublasFP8MMWrapper> cublas_wrapper_;

public:
    ModelParam  model_param_;
    EngineParam engine_param_;
    MoeParam    moe_param_;

#ifdef FUSED_GATED_GEMM
    GemmSwigluRunnerPtr               m_fusedffn_gemm_runner;
    GemmSwigluProfilerPtr             m_fusedffn_gemm_profiler;
    turbomind::QuantMode              m_fusedffn_quant_mode;
    tensorrt_llm::plugins::GemmDims   m_fusedffn_dims{};
    tensorrt_llm::plugins::GemmIdCore m_fusedffn_gemm_id{};
    GemmPluginProfilerManager         m_profiler_manager;
#endif

public:
#ifdef FUSED_MOE_FFN_GEMM
    MoEGemmSwigluRunnerPtr          m_moe_gemm_runner{};
    MOEParallelismConfig            m_parallelism_config;
    MixtureOfExpertsGemmProfilerPtr m_moe_gemm_profiler;
    MOEGemmPluginProfilerManager    m_moe_profiler_manager;

    turbomind::QuantMode m_quant_mode = turbomind::QuantMode::fromDescription();

    tlkc::ActivationType m_activation_type = tlkc::ActivationType::Swiglu;

    tensorrt_llm::plugins::GemmDims  mDims{};
    tensorrt_llm::plugins::GemmIDMoe mGemmId1{};
    tensorrt_llm::plugins::GemmIDMoe mGemmId2{};

    struct WorkspaceInfo {
        void*  workspace{};
        void*  scale_probs{};
        void*  fc2_output{};
        void*  src_to_dest_map{};
        void*  selected_experts{};
        void*  lora_workspace{};
        void*  softmax_tmp_workspace{};
        size_t size{};
    };

    WorkspaceInfo m_fused_moe_workspace;
#endif
};

LlamaLinear::LlamaLinear(cudaStream_t stream, const ModelParam& model, const EngineParam& engine, const MoeParam& moe):
    impl_{std::make_shared<Impl>(stream, model, engine, moe)}
{
}

Tensor LlamaLinear::forward(const Tensor&           input,  //
                            const LlamaDenseWeight& dense,
                            Type                    type,
                            std::optional<Tensor>   output,
                            void*                   param,
                            cudaStream_t            cur_stream)
{
    ssize_t output_dim = type == kFusedSiluFfn ? dense.output_dim / 2 : dense.output_dim;

    Tensor in = input.view({-1, input.shape(-1)});
    Tensor out;

    if (output) {
        out = output->view({in.shape(0), output_dim});
    }
    else {
        out = Tensor({in.shape(0), output_dim}, input.dtype(), input.device());
    }

    impl_->forward(out, in, dense, type, param, cur_stream);

    auto shape   = input.shape();
    shape.back() = out.shape(-1);

    return out.view(shape);
}

void LlamaLinear::forward_moe(Tensor&                 output,
                              const Tensor&           input,
                              const int*              indexes,
                              const int*              offsets,
                              const LlamaDenseWeight& dense,
                              Type                    type,
                              gemm::Context*          context)
{
    if (dense.weight_type != kFloat8_e4m3) {
        impl_->forward_moe(output, input, indexes, offsets, dense, type, context);
    }
    else {
        impl_->forwardFp8(output, input, dense, type, offsets, indexes);
    }
}

void LlamaLinear::forward_cutlass_moe(Tensor&               output,
                                      Tensor&               input,
                                      Tensor&               logits,
                                      //Tensor&               inter_buf_fp8,
                                      Tensor&               cutlass_inout_buf,
                                      const LlamaFfnWeight& weights,
                                      int                   tokens,
                                      int                   expert_num,
                                      bool                  use_shared_stream,
                                      cudaEvent_t           shared_expert_event,
                                      cudaStream_t          shared_expert_stream)
{
    impl_->onellmFP8MoE(output,
                        input,
                        logits,
                        //inter_buf_fp8,
                        cutlass_inout_buf,
                        weights,
                        tokens,
                        expert_num,
                        use_shared_stream,
                        shared_expert_event,
                        shared_expert_stream);
}

void LlamaLinear::set_measure(bool measure)
{
    impl_->dispatch_policy_ = measure ? gemm::DispatchPolicy::kMeasure : gemm::DispatchPolicy::kReuse;
}

int LlamaLinear::Export(std::ostream& os)
{
    if (os) {
        return impl_->gemm_.Export(os);
    }
    return 0;
}

int LlamaLinear::Import(std::istream& is)
{
    auto n_records = 0;
    if (is) {
        n_records = impl_->gemm_.Import(is);
    }
    if (n_records) {
        impl_->dispatch_policy_ = gemm::DispatchPolicy::kReuse;
    };
    return n_records;
}

std::vector<int> LlamaLinear::GetTuningSeq() const
{
    return impl_->gemm_.GetTuningSeq();
}

void LlamaLinear::Impl::onellmFP8Dense(
    Tensor& output, const Tensor& input, const LlamaDenseWeight& weight, Type type, void* param, cudaStream_t cur_stream)
{

    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    void*       output_data = output.raw_data();
    const void* input_data  = input.raw_data();
    const void* weight_data = weight.weight.raw_data();

    int m = input.shape(0);
    int n = weight.output_dim;
    int k = weight.input_dim;

    TM_LOG_DEBUG("mnk: %d, %d, %d\n", m, n, k);

    // NOTE(Alan): 调用fused gated gemm
    if (type == kFusedSiluFfn) {

        void*       output_data             = output.raw_data();
        const void* input_data              = input.raw_data();
        const void* weight_data             = weight.weight.raw_data();
        float       host_quant_to_fp8_scale = *reinterpret_cast<float*>(param);

        size_t const workspace_size = m_fusedffn_gemm_runner->getWorkspaceSize(m, n, k);
        auto const   best_tactic    = m_fusedffn_gemm_profiler->getBestConfig(m, m_fusedffn_gemm_id);

        FT_CHECK_WITH_INFO(best_tactic, "No valid GEMM tactic");
        {
            // NOTE(Alan): output, [m, n / 2]
            //             input0, [m, k] // input
            //             input1, [k, n] // weight
            //             input2, [1, n] // bias
            m_fusedffn_gemm_runner->gemm(output_data,
                                         input_data,
                                         weight_data,
                                         nullptr,
                                         m_fusedffn_quant_mode,
                                         m,
                                         n,
                                         k,
                                         *(weight.host_d0_scale.data<float>()),
                                         *(weight.host_d1_scale.data<float>()),
                                         host_quant_to_fp8_scale,
                                         *best_tactic,
                                         reinterpret_cast<char*>(workspace_.cublas),
                                         workspace_size,
                                         cur_stream==nullptr ? stream_ : cur_stream);
        }
        return;
    }

    ///*
#ifdef ENABLE_FP8
    auto invoke = [&](auto t) {
        using T               = decltype(t);
        bool use_cutlass_gemm = true;

        if (use_cutlass_gemm) {
            const float alpha = 1.0f;
            const float beta  = 0.0f;
            if constexpr (std::is_same_v<half, T>
#ifdef ENABLE_BF16
                          || std::is_same_v<bfloat16_t, T>
#endif
            ) {
                cutlass_scaled_mm(reinterpret_cast<T*>(output_data),
                                  (int)1,  // batch_count
                                  (int)m,
                                  (int)n,
                                  (int)k,
                                  (int64_t)0,
                                  (int64_t)0,
                                  (int64_t)0,
                                  &alpha,
                                  &beta,
                                  reinterpret_cast<const __nv_fp8_e4m3*>(input_data),
                                  reinterpret_cast<const __nv_fp8_e4m3*>(weight_data),
                                  weight.input_scale.data<float>(),
                                  weight.weight_scale.data<float>(),
                                  cur_stream==nullptr ? stream_ : cur_stream);
            }
            sync_check_cuda_error();
        }
        else {
            const float alpha = 1.0f;
            const float beta  = 0.0f;
            // NOTE(Alan): FP8 only support TN, A is Kernel
            cublas_wrapper_->Gemm(reinterpret_cast<T*>(output_data),
                                  (int)1,
                                  (int)m,
                                  (int)n,
                                  (int)k,
                                  (int64_t)0,
                                  (int64_t)0,
                                  (int64_t)0,
                                  &alpha,
                                  &beta,
                                  reinterpret_cast<const __nv_fp8_e4m3*>(input_data),
                                  reinterpret_cast<const __nv_fp8_e4m3*>(weight_data),
                                  weight.input_scale.data<float>(),
                                  weight.weight_scale.data<float>(),
                                  cur_stream==nullptr ? stream_ : cur_stream);
        }
        //*/
    };

    TM_DISPATCH_DTYPES(weight.data_type, invoke, half_t, bfloat16_t);
#endif  // #if ENABLE_FP8
}

void LlamaLinear::Impl::tunningFusedGatedGemm()
{
#ifdef FUSED_GATED_GEMM

    std::vector<int> inter_sizes = model_param_.inter_size;

    // Note(meng): if there is no "shared-expert", there is no need to enter the ffn profile.
    if (!std::accumulate(inter_sizes.begin(), inter_sizes.end(), 0LL)) return;
    
    bool m_all_layer_inter_size_equal = true;

    for (auto& inter_size : inter_sizes)
        m_all_layer_inter_size_equal &= (inter_size == inter_sizes[0]);

    assert(m_all_layer_inter_size_equal && "All layer inter_size are not equal!");

    // NOTE(Alan): 考虑TP的情况
    int inter_size   = inter_sizes[0] / engine_param_.mlp_tp_size;
    int hidden_units = model_param_.hidden_units;

    // TODO(Alan): 支持不同模型，并且调用离线tuning
    // NOTE: minM, maxM, maxN, maxK
    m_fusedffn_dims = {1, engine_param_.max_forward_token_num, int(inter_size * 2), int(hidden_units)};
    // NOTE: maxN, maxK, DataType
    m_fusedffn_gemm_id = {int(inter_size * 2), int(hidden_units), turbomind::DataType::kFloat8_e4m3};
    // NOTE(Alan): only support with dtype fp8
    m_fusedffn_gemm_runner   = std::make_shared<tlkc::CutlassFusedGatedGemmRunner<__nv_fp8_e4m3>>();
    m_fusedffn_gemm_profiler = m_profiler_manager.createGemmPluginProfiler(/* inference */ false);

    m_fusedffn_quant_mode = turbomind::QuantMode::fromQuantAlgo("fp8_static");

    // TODO: do gemm tunning
    m_fusedffn_gemm_profiler->profileTactics(
        m_fusedffn_gemm_runner, turbomind::DataType::kFloat8_e4m3, m_fusedffn_dims, m_fusedffn_gemm_id);

#endif
}

void LlamaLinear::Impl::tuningFusedMoEGemm(const ModelParam& model, const EngineParam& engine, const MoeParam& moe)
{
    if(moe.expert_num.size() <= 0) return;

#ifdef FUSED_MOE_FFN_GEMM
    m_parallelism_config =
        MOEParallelismConfig(engine.moe_tp_size, engine.moe_tp_rank, engine.moe_ep_size, engine.moe_ep_rank);

    auto invoke = [&](auto t) {
        using T = decltype(t);

        if (model.quant_mode.isW4A8AWQ()) {
            m_moe_gemm_runner = std::make_shared<tlkc::CutlassMoeFCRunner<__nv_fp8_e4m3, cutlass::uint4b_t, T, T>>();
        }
        else if (model.quant_mode.isFP8Static()) {
            m_moe_gemm_runner =
                std::make_shared<tlkc::CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp8_e4m3, T, __nv_fp8_e4m3>>();
        }
    };
    TM_DISPATCH_DTYPES(model.data_type, invoke, half_t, bfloat16_t);

    auto setupWorkspace = [&](int64_t num_tokens) -> WorkspaceInfo {
        FT_CHECK_WITH_INFO(model.quant_mode.isFP8Static() || model.quant_mode.isW4A8AWQ(),
                           "Only in FP8 PerTensor Static or W4A8AWQ Mode, we need to setupWorkspace.");

        size_t dtype_size = turbomind::byte_size(model.data_type);

        const int max_expert_num = *std::max_element(moe.expert_num.begin(), moe.expert_num.end()) * engine.moe_ep_size;

        int inter_size = moe.inter_size / engine.moe_tp_size;

        bool has_pre_quant = model.quant_mode.isW4A8AWQ();

        size_t moe_workspace_size = m_moe_gemm_runner->getWorkspaceSize(num_tokens,
                                                                        model.hidden_units,
                                                                        inter_size,
                                                                        max_expert_num,
                                                                        moe.experts_per_token,
                                                                        m_activation_type,
                                                                        m_parallelism_config,
                                                                        /*hasLora()*/ false,
                                                                        /*use_deepseek_fp8_block_scale=*/false,
                                                                        /*min_latency_mode=*/false,
                                                                        has_pre_quant);

        // printf(
        //     "getWorkspaceSize: param: num_tokens=%d, hidden_dim_=%d, param_.inter_size=%d, param_.expert_num=%d,
        //     param_.experts_per_token=%d, moe_workspace_size=%zu\n", num_tokens, hidden_dim_, param_.inter_size,
        //     param_.expert_num,
        //     param_.experts_per_token,
        //     moe_workspace_size);

        // Output of post-softmax routing probabilities
        size_t scale_probabilities_size = num_tokens * max_expert_num * sizeof(float);

        // Permutation map
        size_t src_to_dest_map_size = moe.experts_per_token * num_tokens * sizeof(int);

        // Selected expert map
        size_t selected_expert_size = moe.experts_per_token * num_tokens * sizeof(int);

        size_t lora_workspace_size = 0;
        if (false /*hasLora()*/)  // cant support lora in Moe!
        {
            // int64_t num_reqs_lora = std::min(num_tokens * mK, static_cast<int64_t>(num_reqs * mNumExperts));
            // lora_workspace_size = std::max(mLoraImpl1->getWorkspaceSize(num_tokens * mK, num_reqs_lora,
            // mLoraType),
            //     mLoraImpl2->getWorkspaceSize(num_tokens * mK, num_reqs_lora, mLoraType));
        }

        // Softmax tmp result
        size_t softmax_tmp_size = 0;

        bool const is_pow_2 = (max_expert_num != 0) && ((max_expert_num & (max_expert_num - 1)) == 0);

        if (!is_pow_2 || max_expert_num > 256) {
            softmax_tmp_size = max_expert_num * num_tokens * sizeof(float);
        }

        std::vector<size_t> workspaces{moe_workspace_size,
                                       scale_probabilities_size,
                                       src_to_dest_map_size,
                                       selected_expert_size,
                                       lora_workspace_size,
                                       softmax_tmp_size};

        WorkspaceInfo info{};
        info.size = tensorrt_llm::common::calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());

        if (info.size > workspace_.cutlass_size)
            workspace_.cutlass_size = info.size;

        check_cuda_error(cudaMallocAsync(&workspace_.cutlass, workspace_.cutlass_size, stream_));

        FT_CHECK_WITH_INFO(info.size <= workspace_.cutlass_size,
                           "fused moe ffn gemm workspace size greater than prealloc workspace");

        info.workspace   = workspace_.cutlass;
        info.scale_probs = tensorrt_llm::common::nextWorkspacePtr((int8_t*)info.workspace, moe_workspace_size);
        info.src_to_dest_map =
            tensorrt_llm::common::nextWorkspacePtr((int8_t*)info.scale_probs, scale_probabilities_size);
        info.selected_experts =
            tensorrt_llm::common::nextWorkspacePtr((int8_t*)info.src_to_dest_map, src_to_dest_map_size);
        info.lora_workspace =
            tensorrt_llm::common::nextWorkspacePtr((int8_t*)info.selected_experts, selected_expert_size);
        info.softmax_tmp_workspace =
            tensorrt_llm::common::nextWorkspacePtr((int8_t*)info.lora_workspace, lora_workspace_size);
        return info;
    };

    m_fused_moe_workspace = setupWorkspace(engine.max_forward_token_num);
#endif
}

void LlamaLinear::Impl::onellmFP8MoE(Tensor&               output,
                                     Tensor&               input,
                                     Tensor&               logits,
                                     //Tensor&               inter_buf_fp8,
                                     Tensor&               cutlass_inout_buf,
                                     const LlamaFfnWeight& weights,
                                     int                   tokens,
                                     int                   expert_num,
                                     bool                  use_shared_stream,
                                     cudaEvent_t           shared_expert_event,
                                     cudaStream_t          shared_expert_stream)
{

    // 1. static_cast<const __nv_fp8_e4m3*>(inter_buf_fp8_),
    // 2. logits_
    // 3. cutlass_inout_buf_

#ifdef FUSED_MOE_FFN_GEMM

    int num_experts_per_node = expert_num / m_parallelism_config.ep_size;
    int start_expert         = num_experts_per_node * m_parallelism_config.ep_rank;
    int end_expert           = start_expert + num_experts_per_node;

    num_experts_per_node = expert_num;
    start_expert         = 0;
    end_expert           = expert_num;

    invokeMoESelectExpertAndFinalScales(logits.data<float>(),
                                        static_cast<float*>(m_fused_moe_workspace.scale_probs),
                                        static_cast<int*>(m_fused_moe_workspace.selected_experts),
                                        static_cast<float*>(m_fused_moe_workspace.softmax_tmp_workspace),
                                        tokens,
                                        num_experts_per_node,
                                        moe_param_.experts_per_token,
                                        start_expert,
                                        end_expert,
                                        stream_);

    // if (!isTuning()) {

    //     printf("=== Debug begin log info with logits: %d\n", m_parallelism_config.ep_rank);
    //     print_to_screen<float>(logits.data<float>(), tokens);
    //     printf("=== Debug   end log info with logits: %d\n", m_parallelism_config.ep_rank);

    //     printf("=== Debug begin log info with selected_experts: %d\n", m_parallelism_config.ep_rank);
    //     print_to_screen<int>(static_cast<int*>(m_fused_moe_workspace.selected_experts),
    //                          tokens * moe_param_.experts_per_token);
    //     printf("=== Debug   end log info with selected_experts: %d\n", m_parallelism_config.ep_rank);

    //     printf("=== Debug begin log info with scale_probs: %d\n", m_parallelism_config.ep_rank);
    //     print_to_screen<float>(static_cast<float*>(m_fused_moe_workspace.scale_probs),
    //                            tokens * moe_param_.experts_per_token);
    //     printf("=== Debug   end log info with scale_probs: %d\n", m_parallelism_config.ep_rank);
    // }

    tlkc::QuantParams quant_params{};
    if (weights.quant_mode.isFP8Static()) {
        quant_params = tlkc::QuantParams::FP8(weights.fused_gating_intermediate.d0_scale.data<float>(),
                                              weights.output.d1_scale.data<float>(),
                                              weights.output.d0_scale.data<float>(),
                                              nullptr,   // dont has fp8 final output
                                              nullptr);  // dont has lora
    }
    else if (weights.quant_mode.isW4A8AWQ()) {
        // Note(meng): we only support group-wise w4afp8 quant now
        if (model_param_.group_size > 0) {
            quant_params = tlkc::QuantParams::GroupWise(
                model_param_.group_size,
                weights.fused_gating_intermediate.w_group_quant_scale.raw_data(),  // w1w3 weight quant scale
                weights.output.w_group_quant_scale.raw_data(),                     // w2 weight quant scale
                weights.fused_gating_intermediate.pre_quant_scale.raw_data(),      // w1w3 expert pre_quant scale
                weights.output.pre_quant_scale.raw_data(),                         // w2 expert pre_quant scale
                nullptr,                                                           // w1w3 expert zeros, no zeros
                nullptr,                                                           // w2 expert zeros, no zeros
                weights.fused_gating_intermediate.d0_scale.data<float>(),          // w1w3 group_wise fp8 alpha
                weights.output.d0_scale.data<float>());                            // w2 group_wise fp8 alpha
        }
    }

    float m_sparse_mixer_epsilon = -INFINITY;

    tensorrt_llm::kernels::LoraParams lora_params{};
    tlkc::MoeMinLatencyParams min_latency_params{};

    // tkc::CutlassGemmConfig gemm1_config(tkc::CutlassTileConfigSM90::CtaShape128x16x128B,
    //                                     tkc::MainloopScheduleType::AUTO,
    //                                     tkc::EpilogueScheduleType::AUTO,
    //                                     tkc::ClusterShape::ClusterShape_2x1x1);
    // tkc::CutlassGemmConfig gemm2_config(tkc::CutlassTileConfigSM90::CtaShape128x16x128B,
    //                                     tkc::MainloopScheduleType::AUTO,
    //                                     tkc::EpilogueScheduleType::AUTO,
    //                                     tkc::ClusterShape::ClusterShape_2x1x1);
    // m_moe_gemm_runner->setTactic(gemm1_config, gemm2_config);

    auto gemm_configs = m_moe_gemm_runner->getDefaultTactics(tokens);
    m_moe_gemm_runner->setTactic(gemm_configs[0], gemm_configs[1]);

    // auto gemm1 = m_gemm_profiler->getBestConfig(tokens, mGemmId1);
    // auto gemm2 = m_gemm_profiler->getBestConfig(tokens, mGemmId2);
    // m_moe_gemm_runner->setTactic(gemm1, gemm2);

    int inter_size = moe_param_.inter_size / engine_param_.moe_tp_size;
    int hidden_dim = model_param_.hidden_units;

    // NOTE(Alan): 当前函数会拆分成多个Kernel
    //             1.  tensorrt_llm::kernels::moeSoftmax
    //             2.  tensorrt_llm::kernels::moeTopK
    //             3.  cub::CUB_200400_900_NS::DeviceRadixSortSingleTileKernel
    //             4.  tensorrt_llm::kernels::computeExpertFirstTokenOffsetKernel
    //             5.  tensorrt_llm::kernels::expandInputRowsKernel
    //             6.  tensorrt_llm::kernels::computeStridesHopperKernel
    //             7.  cutlass::group_gemm
    //             8.  tensorrt_llm::kernels::doActivationKernel
    //             9.  Memset
    //             10. tensorrt_llm::kernels::computeStridesHopperKernel
    //             11. cutlass::group_gemm
    //             12. cutlass::Fused_Gated_GEMM 【shared expert ffn w1w3】
    //             13. cutlass::W2_GEMM 【shared expert ffn w2】
    // clang-format off
    
        void* input_activations = input.raw_data();

        // if (!isTuning()) {
        //     printf("=== Debug begin log info with input: %d\n", m_parallelism_config.ep_rank);
        //     print_to_screen<__nv_fp8_e4m3>(static_cast<__nv_fp8_e4m3*>(input.raw_data()), tokens);
        //     printf("=== Debug   end log info with input: %d\n", m_parallelism_config.ep_rank);
        // }

        m_moe_gemm_runner->runMoe(
            /*void const* input_activations*/ input_activations,
            /*void const* input_sf*/ nullptr,
            /*int const* token_selected_experts*/ static_cast<int const*>(m_fused_moe_workspace.selected_experts),
            /*float const* token_final_scales*/ static_cast<float const*>(m_fused_moe_workspace.scale_probs),
            /*void const* fc1_expert_weights*/ weights.fused_gating_intermediate.weight.raw_data(),
            /*void const* fc1_expert_biases*/ weights.fused_gating_intermediate.bias.data_or((void*)nullptr),
            /*ActivationType fc1_activation_type*/ m_activation_type,
            /*void const* fc2_expert_weights*/ weights.output.weight.raw_data(),
            /*void const* fc2_expert_biases*/ weights.output.bias.data_or((void*)nullptr),
            /*QuantParams quant_params*/ quant_params,
            /*int64_t const num_rows*/ tokens,
            /*int64_t const hidden_size*/ hidden_dim,
            /*int64_t const inter_size*/ inter_size,
            /*int const num_experts*/ expert_num,
            /*int const experts_per_token*/ moe_param_.experts_per_token,
            /*char* workspace_ptr*/ static_cast<char*>(m_fused_moe_workspace.workspace),

            // Outputs
            /*void* final_output*/ use_shared_stream ? output.raw_data() : cutlass_inout_buf.raw_data(),
            /*int* expanded_source_row_to_expanded_dest_row*/ static_cast<int*>(m_fused_moe_workspace.src_to_dest_map),
            /*MOEParallelismConfig parallelism_config*/ m_parallelism_config,
            /*bool use_lora*/ false,
            /*LoraParams& lora_params*/ lora_params,
            /*bool use_deepseek_fp8_block_scale*/ false,
            /*bool min_latency_mode*/ false,
            /*MoeMinLatencyParams& min_latency_params*/ min_latency_params,
            /* cudaStream_t                    */stream_,
            /* bool                            */use_shared_stream,
            /* cudaEvent_t                     */shared_expert_event,
            /* cudaStream_t                    */shared_expert_stream);
        // clang-format on

        // if (!isTuning()) {
        //     printf("=== Debug begin log info with output: %d\n", m_parallelism_config.ep_rank);
        //     print_to_screen<__nv_bfloat16>(static_cast<__nv_bfloat16*>(cutlass_inout_buf.raw_data()), tokens);
        //     printf("=== Debug   end log info with output: %d\n", m_parallelism_config.ep_rank);
        // }
#endif
}

}  // namespace turbomind
