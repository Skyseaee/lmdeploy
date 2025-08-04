// clang-format will break include orders
// clang-format off
#include <cudaTypedefs.h>

#if defined CUDA_VERSION && CUDA_VERSION >= 12000

//#include <torch/all.h>

//#include <ATen/cuda/CUDAContext.h>

#include <iostream>
#include <sstream>
#include <vector>

#ifdef CUTLASS_FP8

#include <cutlass/cutlass.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cutlass/numeric_types.h>

#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/util/device_memory.h>
#include <cutlass/epilogue/thread/activation.h>

#include "broadcast_load_epilogue_c3x.hpp"
#include "common.hpp"
// clang-format on

using namespace cute;

/*
   This file defines quantized GEMM operations using the CUTLASS 3.x API, for
   NVIDIA GPUs with sm90a (Hopper) or later.

   Epilogue functions can be defined to post-process the output before it is
   written to GPU memory.
   Epilogues must contain a public type named EVTCompute of type Sm90EVT,
   as well as a static prepare_args function that constructs an
   EVTCompute::Arguments struct.
*/

namespace {

// A wrapper for the GEMM kernel that is used to guard against compilation on
// architectures that will never use the kernel. The purpose of this is to
// reduce the size of the compiled binary.
// __CUDA_ARCH__ is not defined in host code, so this lets us smuggle the ifdef
// into code that will be executed on the device where it is defined.
template <typename Kernel>
struct enable_sm90_or_later : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
  #if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 900
    Kernel::operator()(std::forward<Args>(args)...);
  #endif
  }
};

/*
 * This class provides the common load descriptors for the
 * ScaledEpilogue[...] classes
 */
template <typename ElementAcc, typename ElementD, typename EpilogueDescriptor>
struct ScaledEpilogueBase {
 protected:
  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

  template <typename T>
  using ColOrScalarLoad = cutlass::epilogue::fusion::Sm90ColOrScalarBroadcast<
      0 /*Stages*/, typename EpilogueDescriptor::TileShape, T,
      Stride<Int<1>, Int<0>, Int<0>>>;

  template <typename T>
  using RowOrScalarLoad = cutlass::epilogue::fusion::Sm90RowOrScalarBroadcast<
      0 /*Stages*/, typename EpilogueDescriptor::TileShape, T,
      Stride<Int<0>, Int<1>, Int<0>>>;

  // Don't want to support nullptr by default
  template <typename T, bool EnableNullPtr = false>
  using ColLoad = cutlass::epilogue::fusion::Sm90ColBroadcast<
      0 /*Stages*/, typename EpilogueDescriptor::TileShape, T, T,
      Stride<Int<1>, Int<0>, Int<0>> , 128 / sizeof_bits_v<T>, EnableNullPtr>;

  // Don't want to support nullptr by default
  template <typename T, bool EnableNullPtr = false>
  using RowLoad = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0 /*Stages*/, typename EpilogueDescriptor::TileShape, T, T,
      Stride<Int<0>, Int<1>, Int<0>> , 128 / sizeof_bits_v<T>, EnableNullPtr>;

  // This utility function constructs the arguments for the load descriptors
  // from a tensor. It can handle both row and column, as well as row/column or
  // scalar cases.
  template <typename Descriptor, typename T>
  static auto args_from_tensor(const T* tensor, const int tensor_num) {
    using Arguments = typename Descriptor::Arguments;
    //auto* data_ptr = static_cast<T*>(tensor.data_ptr());
    
    if constexpr (std::is_same_v<Descriptor, ColOrScalarLoad<T>> ||
                  std::is_same_v<Descriptor, RowOrScalarLoad<T>>) {
      return Arguments{tensor, tensor_num != 1};
    } else {
      static_assert(!std::is_same_v<Descriptor, ColLoad<T, true>> &&
                    !std::is_same_v<Descriptor, RowLoad<T, true>>);
      return Arguments{tensor};
    }
  }

  // This overload handles the case where there might not be a tensor, in which
  // case a nullptr is passed and a constant (0) is used.
  template <typename Descriptor, typename T>
  static auto args_from_tensor(T* tensor) {
    using Arguments = typename Descriptor::Arguments;
    //auto* data_ptr = tensor ? static_cast<T*>(tensor->data_ptr()) : nullptr;
    static_assert(std::is_same_v<Descriptor, ColLoad<T, true>> ||
                  std::is_same_v<Descriptor, RowLoad<T, true>>);
    return Arguments{tensor};
  }
};

/*
   This epilogue function defines a quantized GEMM operation similar to
   torch.scaled_mm_.

   A and B may be both either int8 or fp8_e4m3. A can be
   quantized per-tensor or per-row. B can be quantized per-tensor or per-column.
   Any combination of per-tensor and per-row or column is supported.
   A and B must have symmetric quantization (zero point == 0).

   So the GEMM operation is D = (a_scales * A) (b_scales * B), where the
   scales are applied elementwise with numpy-style broadcasting.

   ScaleA and ScaleB define the epilogue functions that apply the scales for
   the A and B operands respectively. These scales may be either per-tensor or
   per row or column.
*/
template <typename ElementAcc, typename ElementD, typename EpilogueDescriptor>
struct ScaledEpilogue
    : private ScaledEpilogueBase<ElementAcc, ElementD, EpilogueDescriptor> {
 private:
  using SUPER = ScaledEpilogueBase<ElementAcc, ElementD, EpilogueDescriptor>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;

  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 =
      cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;

  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute =
      cutlass::epilogue::fusion::Sm90EVT<Compute1, ScaleA, EVTCompute0>;
  using ArgumentType = typename EVTCompute::Arguments;

  static ArgumentType prepare_args(const float* a_scales,
                                   const float* b_scales) {
    // auto a_args = SUPER::template args_from_tensor<ScaleA, float>(a_scales, 1);
    // auto b_args = SUPER::template args_from_tensor<ScaleB, float>(b_scales, 1);

    // //typename EVTCompute0::Arguments evt0_args{b_args};
    // //return ArgumentType{a_args, b_args};
    // return ArgumentType{};

    auto a_args = SUPER::template args_from_tensor<ScaleA, float>(a_scales, 1);
    auto b_args = SUPER::template args_from_tensor<ScaleB, float>(b_scales, 1);

    typename EVTCompute0::Arguments evt0_args{b_args, {}, {}};
    return ArgumentType{a_args, evt0_args, {}};
  }
};


/*
   This epilogue function defines a quantized GEMM operation similar to
   torch.scaled_mm_.

   A and B may be both either int8 or fp8_e4m3. A can be
   quantized per-tensor or per-row. B can be quantized per-tensor or per-column.
   Any combination of per-tensor and per-row or column is supported.
   A and B must have symmetric quantization (zero point == 0).

   So the GEMM operation is D = (a_scales * A) (b_scales * B), where the
   scales are applied elementwise with numpy-style broadcasting.

   ScaleA and ScaleB define the epilogue functions that apply the scales for
   the A and B operands respectively. These scales may be either per-tensor or
   per row or column.
*/
template <typename ElementAcc, typename ElementD, typename EpilogueDescriptor>
struct ScaledEpilogueAct
    : private ScaledEpilogueBase<ElementAcc, ElementD, EpilogueDescriptor> {
 private:
  using SUPER = ScaledEpilogueBase<ElementAcc, ElementD, EpilogueDescriptor>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;

  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 =
      cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;

  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute1 =
      cutlass::epilogue::fusion::Sm90EVT<Compute1, ScaleA, EVTCompute0>;

  using Compute2 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::epilogue::thread::SiLu, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

 public:

  using EVTCompute = cutlass::epilogue::fusion::Sm90EVT<Compute2, EVTCompute1>;

  using ArgumentType = typename EVTCompute::Arguments;

  static ArgumentType prepare_args(const float* a_scales,
                                   const float* b_scales) {
    auto a_args = SUPER::template args_from_tensor<ScaleA, float>(a_scales, 1);
    auto b_args = SUPER::template args_from_tensor<ScaleB, float>(b_scales, 1);

    typename EVTCompute0::Arguments evt0_args{b_args};
    return ArgumentType{a_args, evt0_args};
  }
};

template <typename ElementAB_, typename ElementD_,
          template <typename, typename, typename> typename Epilogue_,
          typename TileShape, typename ClusterShape, typename KernelSchedule,
          typename EpilogueSchedule>
struct cutlass_3x_gemm {
  using ElementAB = ElementAB_;
  using ElementD = ElementD_;
  using ElementAcc =
      typename std::conditional<std::is_same_v<ElementAB, int8_t>, int32_t,
                                float>::type;

  using EpilogueDescriptor =
      cutlass::epilogue::collective::detail::EpilogueDescriptor<
          TileShape, cutlass::epilogue::collective::EpilogueTileAuto, ElementD,
          ElementD, EpilogueSchedule>;

  using Epilogue = Epilogue_<ElementAcc, ElementD, EpilogueDescriptor>;

  using StrideD = Stride<int64_t, Int<1>, Int<0>>;
  using ElementC = void;
  using StrideC = StrideD;

  using EVTCompute = typename Epilogue::EVTCompute;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape,
          ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAcc, float, ElementC, StrideC, 4, ElementD, StrideD, 4,
          EpilogueSchedule, EVTCompute>::CollectiveOp;

  static constexpr size_t CEStorageSize =
      sizeof(typename CollectiveEpilogue::SharedStorage);
  using Stages = typename cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(CEStorageSize)>;

  // clang-format off
  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, 
          ElementAB, cutlass::layout::RowMajor, 16, 
          ElementAB, cutlass::layout::ColumnMajor, 16, 
          ElementAcc, TileShape, ClusterShape,
          Stages,
          KernelSchedule>::CollectiveOp;
  // clang-format on

  using KernelType = enable_sm90_or_later<cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>>;

  struct GemmKernel : public KernelType {};
};

template <typename Gemm, typename T, typename... EpilogueArgs>
void cutlass_gemm_caller(T*       res,
                         int                  batchCount,
                         int                  m,
                         int                  n,
                         int                  k,
                         int64_t              stride_a,
                         int64_t              stride_b,
                         int64_t              stride_d,
                         const float*         alpha,
                         const float*         beta,
                         const __nv_fp8_e4m3* input,
                         const __nv_fp8_e4m3* kernel,
                         const float*         input_scale,
                         const float*         kernel_scale,
                         cudaStream_t         stream,
                         EpilogueArgs&&... epilogue_params) {
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;

  const int64_t lda = k;
  const int64_t ldb = k;
  const int64_t ldd = n;

  using StrideA = Stride<int64_t, Int<1>, int64_t>;
  using StrideB = Stride<int64_t, Int<1>, int64_t>;
  using StrideC = typename Gemm::StrideC;

  StrideA a_stride{lda, Int<1>{}, 0};
  StrideB b_stride{ldb, Int<1>{}, 0};
  StrideC c_stride{ldd, Int<1>{}, Int<0>{}};

  using GemmKernel = typename Gemm::GemmKernel;
  typename GemmKernel::ProblemShape prob_shape{m, n, k, 1};

  auto a_ptr = reinterpret_cast<const ElementAB*>(input);
  auto b_ptr = reinterpret_cast<const ElementAB*>(kernel);
  typename GemmKernel::MainloopArguments mainloop_args{a_ptr, a_stride, b_ptr,
                                                       b_stride};
  
  auto c_ptr = reinterpret_cast<const ElementD*>(res);
  typename GemmKernel::EpilogueArguments epilogue_args{
      Gemm::Epilogue::prepare_args(input_scale, kernel_scale),
          //std::forward<EpilogueArgs>(epilogue_params)...),
          c_ptr, c_stride, c_ptr, c_stride};

  typename GemmKernel::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                      prob_shape, mainloop_args, epilogue_args};

  // Launch the CUTLASS GEMM kernel.
  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  GemmOp gemm_op;

  CUTLASS_CHECK(gemm_op.can_implement(args));

  size_t workspace_size = gemm_op.get_workspace_size(args);
  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  CUTLASS_CHECK(gemm_op.run(args, workspace.get(), stream));
}

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_default {
  // M in (128, inf)
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using Cutlass3xGemm =
      cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                      KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_M128 {
  // M in (64, 128]
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using Cutlass3xGemm =
      cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                      KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_M64 {
  // M in [1, 64]
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _64, _128>;
  using ClusterShape = Shape<_1, _8, _1>;

  using Cutlass3xGemm =
      cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                      KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_default {
  // For M > 128 and any N
  static_assert(std::is_same<InType, int8_t>());
  using KernelSchedule =
      typename cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using Cutlass3xGemm =
      cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                      KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_M128 {
  // For M in (64, 128] and any N
  static_assert(std::is_same<InType, int8_t>());
  using KernelSchedule =
      typename cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using Cutlass3xGemm =
      cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                      KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_M64 {
  // For M in (32, 64] and any N
  static_assert(std::is_same<InType, int8_t>());
  using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _64, _256>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Cutlass3xGemm =
      cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                      KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_M32_NBig {
  // For M in [1, 32] and N >= 8192
  static_assert(std::is_same<InType, int8_t>());
  using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _128, _256>;
  using ClusterShape = Shape<_1, _4, _1>;
  using Cutlass3xGemm =
      cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                      KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_M32_NSmall {
  // For M in [1, 32] and N < 8192
  static_assert(std::is_same<InType, int8_t>());
  using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _64, _256>;
  using ClusterShape = Shape<_1, _8, _1>;
  using Cutlass3xGemm =
      cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                      KernelSchedule, EpilogueSchedule>;
};

}  // namespace

template <typename InType, typename OutType, typename T,
          template <typename, typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_gemm_sm90_fp8_dispatch(T*       res,
                                    int                  batchCount,
                                    int                  m,
                                    int                  n,
                                    int                  k,
                                    int64_t              stride_a,
                                    int64_t              stride_b,
                                    int64_t              stride_d,
                                    const float*         alpha,
                                    const float*         beta,
                                    const __nv_fp8_e4m3* input,
                                    const __nv_fp8_e4m3* kernel,
                                    const float*         input_scale,
                                    const float*         kernel_scale,
                                    cudaStream_t         stream,
                                    EpilogueArgs&&... args) {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  
  // TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
  // TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);

  using Cutlass3xGemmDefault =
      typename sm90_fp8_config_default<InType, OutType,
                                       Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemmM64 =
      typename sm90_fp8_config_M64<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemmM128 =
      typename sm90_fp8_config_M128<InType, OutType, Epilogue>::Cutlass3xGemm;

  //uint32_t const m = a.size(0);
  uint32_t const mp2 =
      std::max(static_cast<uint32_t>(64), next_pow_2(m));  // next power of 2

  if (mp2 <= 64) {
    // m in [1, 64]
    return cutlass_gemm_caller<Cutlass3xGemmM64>(
        res, batchCount, m, n, k, stride_a, stride_b, stride_d, alpha, beta, input, kernel, input_scale, kernel_scale, stream, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 128) {
    // m in (64, 128]
    return cutlass_gemm_caller<Cutlass3xGemmM128>(
      res, batchCount, m, n, k, stride_a, stride_b, stride_d, alpha, beta, input, kernel, input_scale, kernel_scale, stream, std::forward<EpilogueArgs>(args)...);
  } else {
    // m in (128, inf)
    return cutlass_gemm_caller<Cutlass3xGemmDefault>(
      res, batchCount, m, n, k, stride_a, stride_b, stride_d, alpha, beta, input, kernel, input_scale, kernel_scale, stream, std::forward<EpilogueArgs>(args)...);
  }
}

template <typename T,
          template <typename, typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_scaled_mm_sm90_epilogue(T*        res,
                                     int                  batchCount,
                                     int                  m,
                                     int                  n,
                                     int                  k,
                                     int64_t              stride_a,
                                     int64_t              stride_b,
                                     int64_t              stride_d,
                                     const float*         alpha,
                                     const float*         beta,
                                     const __nv_fp8_e4m3* input,
                                     const __nv_fp8_e4m3* kernel,
                                     const float*         input_scale,
                                     const float*         kernel_scale,
                                     cudaStream_t         stream,
                                     EpilogueArgs&&... epilogue_args) {
/*    static_assert(std::is_same<T, half>()
#ifdef ENABLE_BF16
    || std::is_same<T, bfloat16_t>()
#endif
    );
*/

    using OutType = typename std::conditional<
    std::is_same<T, half>::value,
    cutlass::half_t, cutlass::bfloat16_t>::type;
    return cutlass_gemm_sm90_fp8_dispatch<cutlass::float_e4m3_t,
                                          OutType, T, Epilogue>(
        res, batchCount, m, n, k, stride_a, stride_b, stride_d, alpha, beta, 
        input, kernel, input_scale, kernel_scale, stream, std::forward<EpilogueArgs>(epilogue_args)...);
}

template<typename T>
void cutlass_scaled_mm_sm90(T*        res,
                             int                  batchCount,
                             int                  m,
                             int                  n,
                             int                  k,
                             int64_t              stride_a,
                             int64_t              stride_b,
                             int64_t              stride_d,
                             const float*         alpha,
                             const float*         beta,
                             const __nv_fp8_e4m3* input,
                             const __nv_fp8_e4m3* kernel,
                             const float*         input_scale,
                             const float*         kernel_scale,
                             cudaStream_t         stream) {
  /*
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);
  if (bias) {
    TORCH_CHECK(bias->dtype() == c.dtype(),
                "currently bias dtype must match output dtype ", c.dtype());
    return cutlass_scaled_mm_sm90_epilogue<ScaledEpilogueBias>(
        c, a, b, a_scales, b_scales, *bias);
  } else {
  */
    return cutlass_scaled_mm_sm90_epilogue<T, ScaledEpilogue>(res, batchCount, m, n, k, stride_a, stride_b, stride_d, alpha, beta, 
                                    input, kernel, input_scale, kernel_scale, stream);
  //}
}

template void cutlass_scaled_mm_sm90(half*                res,
                                     int                  batchCount,
                                     int                  m,
                                     int                  n,
                                     int                  k,
                                     int64_t              stride_a,
                                     int64_t              stride_b,
                                     int64_t              stride_d,
                                     const float*         alpha,
                                     const float*         beta,
                                     const __nv_fp8_e4m3* input,
                                     const __nv_fp8_e4m3* kernel,
                                     const float*         input_scale,
                                     const float*         kernel_scale,
                                     cudaStream_t         stream);

#ifdef ENABLE_BF16
template void cutlass_scaled_mm_sm90(__nv_bfloat16*       res,
                                     int                  batchCount,
                                     int                  m,
                                     int                  n,
                                     int                  k,
                                     int64_t              stride_a,
                                     int64_t              stride_b,
                                     int64_t              stride_d,
                                     const float*         alpha,
                                     const float*         beta,
                                     const __nv_fp8_e4m3* input,
                                     const __nv_fp8_e4m3* kernel,
                                     const float*         input_scale,
                                     const float*         kernel_scale,
                                     cudaStream_t         stream);
#endif

#endif //#ifdef CUTLASS_FP8

#endif //#if defined CUDA_VERSION && CUDA_VERSION >= 12000
