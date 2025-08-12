// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

#ifdef ENABLE_FP8
#include "src/turbomind/utils/cuda_fp8_utils.h"
#endif

namespace turbomind {

template<typename T>
void invokeFusedAddBiasResidualRMSNorm(
    T* residual, T* in_out, const T* bias, const T* scale, float eps, int batch_size, int n_dims, cudaStream_t stream);

#ifdef ENABLE_FP8
// NOTE(Alan): current only support per token quant
template<typename T>
void invokeFusedAddBiasResidualRMSNormAndQuant(
    T* residual, __nv_fp8_e4m3* out_q, __nv_fp8_e4m3* moe_out_q, T* out, T* in, const T* bias, const T* scale, float eps, float qscale, float moe_qscale, int batch_size, int n_dims, cudaStream_t stream);
#endif

template<typename T>
void invokeMask(T* output, const int* mask, int batch_size, int dim, cudaStream_t stream);

}  // namespace turbomind
