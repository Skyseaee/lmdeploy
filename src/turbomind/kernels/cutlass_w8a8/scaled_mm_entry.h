#include <cudaTypedefs.h>

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#ifdef CUTLASS_FP8
template<typename T>
void cutlass_scaled_mm(T*                res,
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
                          const float          scale_output);

#endif  // #ifdef CUTLASS_FP8