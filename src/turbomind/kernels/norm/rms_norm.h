// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

#include "src/turbomind/core/core.h"

namespace turbomind {

void invokeRMSNorm(Tensor& out, const Tensor& x, const Tensor& w, float eps, cudaStream_t st);

#ifdef ENABLE_FP8
void invokeRMSNormAndQuant(Tensor& out, const Tensor& x, const Tensor& w, float eps, float qscale, cudaStream_t st);
#endif

void invokeRMSNormQK(Tensor& x, const Tensor& w, float eps, cudaStream_t st);

template<class T>
void invokeBiasResidualRMSNorm(
    T* residual, T* hidden_states, const T* weights, const T* bias, int dims, int num, float eps, cudaStream_t st);

void invokeResidualBiasRMSNorm(void*        hidden_states,
                               void*        residual,
                               const void*  weights,
                               const void*  bias,
                               DataType     dtype,
                               int          dims,
                               int          num,
                               float        eps,
                               cudaStream_t st);

void invokeResidualBiasRMSNormAndQuantFP8(void*        hidden_states_fp8,
                                          void*        residual,
                                          void*        moe_fp8_buf,
                                          void*        moe_fp16_buf,
                                          void*        hidden_states,
                                          const void*  weights,
                                          const void*  bias,
                                          const float  shared_expert_scale,
                                          const float  moe_expert_scale,
                                          DataType     dtype,
                                          int          dims,
                                          int          num,
                                          float        eps,
                                          cudaStream_t st);

}  // namespace turbomind
