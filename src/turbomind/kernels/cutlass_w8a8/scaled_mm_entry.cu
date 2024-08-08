#include "src/turbomind/utils/logger.h"
#include "scaled_mm_entry.h"

#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/memory_utils.h"
#include "cutlass/arch/mma.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/error_metrics.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#ifdef CUTLASS_FP8

#if defined CUDA_VERSION && CUDA_VERSION >= 12000
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
                             cudaStream_t         stream);
#endif

bool cutlass_scaled_mm_supports_fp8(int64_t cuda_device_capability) {
  // CUTLASS FP8 kernels need at least
  //   CUDA 12.0 on SM90 systems (Hopper)
  //   CUDA 12.4 on SM89 systems (Lovelace)

#if defined CUDA_VERSION
  if (cuda_device_capability >= 90) {
    return CUDA_VERSION >= 12000;
  } else if (cuda_device_capability >= 89) {
    return CUDA_VERSION >= 12040;
  }
#endif

  return false;
}

int32_t get_sm_version_num() {
  int32_t major_capability, minor_capability;
  cudaDeviceGetAttribute(&major_capability, cudaDevAttrComputeCapabilityMajor,
                         0);
  cudaDeviceGetAttribute(&minor_capability, cudaDevAttrComputeCapabilityMinor,
                         0);
  int32_t version_num = major_capability * 10 + minor_capability;
  return version_num;
}

template<typename T>
void cutlass_scaled_mm(T*        res,
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
  // Checks for conformality
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2);
  TORCH_CHECK(c.size(0) == a.size(0) && a.size(1) == b.size(0) &&
              b.size(1) == c.size(1));
  TORCH_CHECK(a_scales.numel() == 1 || a_scales.numel() == a.size(0));
  TORCH_CHECK(b_scales.numel() == 1 || b_scales.numel() == b.size(1));

  // Check for strides and alignment
  TORCH_CHECK(a.stride(1) == 1 && c.stride(1) == 1);  // Row-major
  TORCH_CHECK(b.stride(0) == 1);                      // Column-major
  TORCH_CHECK(c.stride(0) % 16 == 0 &&
              b.stride(1) % 16 == 0);  // 16 Byte Alignment
  TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());

  if (bias) {
    TORCH_CHECK(bias->numel() == b.size(1) && bias->is_contiguous() &&
                bias->dim() == 1);
  }

  at::cuda::OptionalCUDAGuard const device_guard(device_of(a));
  */

  int32_t version_num = get_sm_version_num();
  if (version_num >= 90) {
    // Hopper

    // Guard against compilation issues for sm90 kernels
#if defined CUDA_VERSION && CUDA_VERSION >= 12000
    cutlass_scaled_mm_sm90(res, batchCount, m, n, k, stride_a, stride_b, stride_d, alpha, beta, 
      input, kernel, input_scale, kernel_scale, stream);
#else
    TM_LOG_INFO("[Cutlass_W8A8] CUDA_VERSION %d not support.", CUDA_VERSION);
    //cutlass_scaled_mm_sm80(c, a, b, a_scales, b_scales, bias);
#endif
  } else if (version_num == 89) {
    // Ada Lovelace
    TM_LOG_INFO("[Cutlass_W8A8] sm89 not support.");
    //cutlass_scaled_mm_sm89(c, a, b, a_scales, b_scales, bias);
  } else if (version_num >= 80) {
    // Ampere
    TM_LOG_INFO("[Cutlass_W8A8] sm80 not support.");
    //cutlass_scaled_mm_sm80(c, a, b, a_scales, b_scales, bias);
  } else {
    // Turing
    //TORCH_CHECK(version_num >= 75);
    TM_LOG_INFO("[Cutlass_W8A8] sm%d not support.", version_num);
    //cutlass_scaled_mm_sm75(c, a, b, a_scales, b_scales, bias);
  }
}

template <typename ElementT_>
using Activation = cutlass::epilogue::thread::SiLu<ElementT_>;

void fused_gated_gemm_ref(__nv_fp8_e4m3*       res,
                          int                  m,
                          int                  n,
                          int                  k,
                          const __nv_fp8_e4m3* input,
                          const __nv_fp8_e4m3* kernel,
                          const float          alpha,
                          const float          beta,
                          const float          scale_d0,
                          const float          scale_d1,
                          const float          scale_output)
{
  using ElementT = cutlass::float_e4m3_t;
  using ElementAccumulatorT = float;
  using ElementComputeT = float;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementD2x = float;

  // Create instantiation for device reference gemm kernel
  // Reference device GEMM implementation type
  using DeviceGemmReference = cutlass::reference::device::Gemm<ElementT, cutlass::layout::RowMajor, ElementT, LayoutB,
      ElementD2x, cutlass::layout::RowMajor, ElementAccumulatorT, ElementAccumulatorT>;
  DeviceGemmReference gemm_reference;

  auto problem_size = cutlass::gemm::GemmCoord{m, n, k};
  auto problem_size_out = cutlass::gemm::GemmCoord{m, n / 2, k};

  cutlass::HostTensor<ElementT, cutlass::layout::RowMajor> tensor_a;
  cutlass::HostTensor<ElementT, LayoutB> tensor_b;
  cutlass::HostTensor<ElementT, cutlass::layout::RowMajor> tensor_d;
  cutlass::HostTensor<ElementD2x, cutlass::layout::RowMajor> tensor_ref_d_2x;
  cutlass::HostTensor<ElementT, cutlass::layout::RowMajor> tensor_ref_d;
  cutlass::HostTensor<ElementT, cutlass::layout::RowMajor> tensor_c_bias;

  // Initialize tensors using CUTLASS helper functions
  tensor_a.resize(problem_size.mk());          // <- Create matrix A with dimensions M x K
  tensor_b.resize(problem_size.kn());  // <- Create matrix merged B with dimensions K x N
  tensor_c_bias.resize({1, problem_size.n()}); // <- Create broadcast vector with dimensions 1 x N
  tensor_d.resize(problem_size_out.mn()); // <- Create matrix D with dimensions M x N/2 used to store output from CUTLASS kernel
  tensor_ref_d_2x.resize(problem_size.mn()); // <- Create temp matrix D with dimensions M x N used to store output from reference kernel
  tensor_ref_d.resize(problem_size_out.mn()); // <- Create matrix D with dimensions M x N/2 used to store output from reference kernel

  turbomind::cudaD2Hcpy<__nv_fp8_e4m3>(reinterpret_cast<__nv_fp8_e4m3 *>(tensor_a.host_data()), reinterpret_cast<const __nv_fp8_e4m3 *>(input), m * k);
  turbomind::cudaD2Hcpy<__nv_fp8_e4m3>(reinterpret_cast<__nv_fp8_e4m3 *>(tensor_b.host_data()), reinterpret_cast<const __nv_fp8_e4m3 *>(kernel), n * k);
  cutlass::reference::host::TensorFill(tensor_c_bias.host_view());
  cutlass::reference::host::TensorFill(tensor_ref_d_2x.host_view());

  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c_bias.sync_device();
  tensor_ref_d_2x.sync_device();

  // Wait for kernels to finish
  turbomind::check_cuda_error(cudaDeviceSynchronize());
  // Launch device reference gemm kernel
  gemm_reference(
      problem_size, 
      ElementAccumulatorT(alpha),
      tensor_a.device_ref(),
      tensor_b.device_ref(), 
      ElementAccumulatorT(beta),
      tensor_ref_d_2x.device_ref(),
      tensor_ref_d_2x.device_ref()
  );

  // Wait for kernels to finish
  turbomind::check_cuda_error(cudaDeviceSynchronize());

  // Copy output data from reference kernel to host for comparison
  tensor_ref_d_2x.sync_host();

  cutlass::NumericConverter<ElementT, ElementComputeT, cutlass::FloatRoundStyle::round_to_nearest> converter;
  int half_n = problem_size.n() / 2;
  for (int i = 0; i < problem_size.m(); i++)
  {
      for (int j = 0; j < half_n; j++)
      {
          auto s = scale_output
              * ElementComputeT(scale_d0 * tensor_ref_d_2x.host_view().ref().at({i, j}))
              * Activation<ElementComputeT>{}(scale_d1 * tensor_ref_d_2x.at({i, j + half_n}));
          auto t = converter(s);
          tensor_ref_d.host_view().ref().at({i, j}) = t;
      }
  }

  turbomind::check_cuda_error(cudaDeviceSynchronize());
  turbomind::cudaH2Dcpy<__nv_fp8_e4m3>(res, reinterpret_cast<const __nv_fp8_e4m3 *>(tensor_ref_d.host_data()), m * half_n);
  turbomind::check_cuda_error(cudaDeviceSynchronize());
}


template void cutlass_scaled_mm(half*                res,
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
template void cutlass_scaled_mm(__nv_bfloat16*       res,
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