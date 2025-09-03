#include <stddef.h>

#ifdef CUTLASS_FP8

#include "cutlass/cutlass.h"

#include "scaled_mm_c2x.cuh"
#include "scaled_mm_c2x_sm89_fp8_dispatch.cuh"

#include "cutlass_extensions/epilogue/collective/scaled_mm_epilogues_c2x.hpp"

/*
   This file defines quantized GEMM operations using the CUTLASS 2.x API, for
   NVIDIA GPUs with SM versions on sm89 (Ada).
*/

template<typename T,
    template<typename, typename> typename Epilogue, typename... EpilogueArgs>
void cutlass_scaled_mm_sm89_epilogue(T*                   res,
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
                                     EpilogueArgs&&... epilogue_args)
{
/*
    turbomind::FT_CHECK(std::is_same<T, half>()
#ifdef ENABLE_BF16
             || std::is_same<T, bfloat16_t>()
#endif
    )
*/

    return cutlass_gemm_sm89_fp8_dispatch<cutlass::float_e4m3_t, T, Epilogue>(
        res, batchCount, m, n, k, stride_a, stride_b, stride_d, alpha, beta,
        input, kernel, input_scale, kernel_scale, stream, std::forward<EpilogueArgs>(epilogue_args)...);
}

template<typename T>
void cutlass_scaled_mm_sm89(T*                   res,
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
                            cudaStream_t         stream)
{

    // Note(meng): Currently, support for "bias" is not available.
    return cutlass_scaled_mm_sm89_epilogue<T, c2x::ScaledEpilogue>(
        res, batchCount, m, n, k, stride_a, stride_b, stride_d, alpha, beta,
        input, kernel, input_scale, kernel_scale, stream);
}

template void cutlass_scaled_mm_sm89(half*                res,
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
template void cutlass_scaled_mm_sm89(__nv_bfloat16*       res,
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

#endif  // #ifdef CUTLASS_FP8
