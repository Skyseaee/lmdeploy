/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cuda_fp8_utils.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind {
#ifdef ENABLE_FP8

template <typename scalar_t>
struct __align__(8) vec4_t {
  scalar_t x;
  scalar_t y;
  scalar_t z;
  scalar_t w;
};

typedef struct __align__(4) {
  __nv_fp8_e4m3 x;
  __nv_fp8_e4m3 y;
  __nv_fp8_e4m3 z;
  __nv_fp8_e4m3 w;
}
float8x4_t;

// __global__ void PrintFP8Data(__nv_fp8_e4m3* data, const int size)
// {
//     printf("FP8 Data: ");
//     float scale = 1.0;
//     for(int i=0; i<size; i++)
//     {
//         __half dst = scaled_vec_conversion<__half, __nv_fp8_e4m3>(data[i], scale);
//         printf("%f ", __half2float(dst));
//     }
//     printf("\n");
// }

// void invokePrintFP8Data(__nv_fp8_e4m3* data, const int size, cudaStream_t stream)
// {
//     PrintFP8Data<<<1, 1, 0, stream>>>(data, size);
// }

__global__ void half2Fp8(__nv_fp8_e4m3* output, const half* input, int num_element)
{
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_element) {
    output[idx] = vec_conversion<__nv_fp8_e4m3, half>(input[idx]);
  }
}

void invokeHalf2Fp8(__nv_fp8_e4m3* output, const half* input, int num_element) {
  int threads_per_block = 256;
  int num_blocks = (num_element + threads_per_block - 1) / threads_per_block;
  half2Fp8<<<num_blocks, threads_per_block>>>(output, input, num_element);
}

template<typename T_OUT, typename T_IN, QUANTIZE_MODE quantize_mode>
__global__ void quantizeMatrix(T_OUT* output, float const* input_scale, T_IN const* input, uint32_t size, uint32_t n)
{
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += blockDim.x * gridDim.x) {
        if (quantize_mode == QUANTIZE_MODE::PER_CHANNEL) {
            output[i] = T_OUT((float)(input[i]) * __ldg(input_scale + (i % n)));
        }
        else {
            output[i] = T_OUT((float)(input[i]) * __ldg(input_scale));
        }
    }
}

template<typename T_OUT, typename T_IN, QUANTIZE_MODE quantize_mode>
void invokeQuantizeMatrix(
    T_OUT* output, float const* input_scale, T_IN const* input, uint32_t size, uint32_t n, cudaStream_t stream)
{
    dim3 grid(32);
    dim3 block(256);
    quantizeMatrix<T_OUT, T_IN, quantize_mode><<<grid, block, 0, stream>>>(output, input_scale, input, size, n);
}

#define defineinvokeQuantizeMatrix(type_out, type_in, mode)                                                            \
    template void invokeQuantizeMatrix<type_out, type_in, mode>(type_out * output,                                     \
                                                                float const*   input_scale,                            \
                                                                type_in const* input,                                  \
                                                                uint32_t       size,                                   \
                                                                uint32_t       n,                                      \
                                                                cudaStream_t   stream);

defineinvokeQuantizeMatrix(__nv_fp8_e4m3, float, QUANTIZE_MODE::PER_CHANNEL);
defineinvokeQuantizeMatrix(__nv_fp8_e4m3, float, QUANTIZE_MODE::PER_TENSOR);
defineinvokeQuantizeMatrix(__nv_fp8_e4m3, half, QUANTIZE_MODE::PER_CHANNEL);
defineinvokeQuantizeMatrix(__nv_fp8_e4m3, half, QUANTIZE_MODE::PER_TENSOR);
defineinvokeQuantizeMatrix(half, __nv_fp8_e4m3, QUANTIZE_MODE::PER_CHANNEL);
defineinvokeQuantizeMatrix(half, __nv_fp8_e4m3, QUANTIZE_MODE::PER_TENSOR);
defineinvokeQuantizeMatrix(float, __nv_fp8_e4m3, QUANTIZE_MODE::PER_CHANNEL);
defineinvokeQuantizeMatrix(float, __nv_fp8_e4m3, QUANTIZE_MODE::PER_TENSOR);
#ifdef ENABLE_BF16
defineinvokeQuantizeMatrix(__nv_fp8_e4m3, __nv_bfloat16, QUANTIZE_MODE::PER_CHANNEL);
defineinvokeQuantizeMatrix(__nv_fp8_e4m3, __nv_bfloat16, QUANTIZE_MODE::PER_TENSOR);
defineinvokeQuantizeMatrix(__nv_bfloat16, __nv_fp8_e4m3, QUANTIZE_MODE::PER_CHANNEL);
defineinvokeQuantizeMatrix(__nv_bfloat16, __nv_fp8_e4m3, QUANTIZE_MODE::PER_TENSOR);
#endif

template <bool is_scale_inverted>
__device__ __forceinline__ __nv_fp8_e4m3 scaled_fp8_conversion(float const val,
                                                          float const scale) {
  float x = 0.0f;
  if constexpr (is_scale_inverted) {
    x = val * scale;
  } else {
    x = val / scale;
  }

  float r = fmax(-FP8_E4M3_MAX, fmin(x, FP8_E4M3_MAX));
  return static_cast<__nv_fp8_e4m3>(r);
}

template <typename T_IN, bool is_scale_inverted>
__device__ void scaled_fp8_conversion_vec(__nv_fp8_e4m3* __restrict__ out,
                                          T_IN const* __restrict__ input,
                                          float const scale,
                                          int64_t const num_elems,
                                          int const tid, int const step) {
  // Vectorized input/output to better utilize memory bandwidth.
  vec4_t<T_IN> const* vectorized_in =
      reinterpret_cast<vec4_t<T_IN> const*>(input);
  float8x4_t* vectorized_out = reinterpret_cast<float8x4_t*>(out);

  int64_t const num_vec_elems = num_elems >> 2;

#pragma unroll 4
  for (int64_t i = tid; i < num_vec_elems; i += step) {
    vec4_t<T_IN> in_vec = vectorized_in[i];
    float8x4_t out_vec;

    out_vec.x = scaled_fp8_conversion<is_scale_inverted>(
        static_cast<float>(in_vec.x), scale);
    out_vec.y = scaled_fp8_conversion<is_scale_inverted>(
        static_cast<float>(in_vec.y), scale);
    out_vec.z = scaled_fp8_conversion<is_scale_inverted>(
        static_cast<float>(in_vec.z), scale);
    out_vec.w = scaled_fp8_conversion<is_scale_inverted>(
        static_cast<float>(in_vec.w), scale);
    vectorized_out[i] = out_vec;
  }

  // Handle the remaining elements if num_elems is not divisible by 4
  for (int64_t i = num_vec_elems * 4 + tid; i < num_elems; i += step) {
    out[i] = scaled_fp8_conversion<is_scale_inverted>(
        static_cast<float>(input[i]), scale);
  }
}

template <typename T_IN>
__global__ void scaled_fp8_quant_kernel(__nv_fp8_e4m3* __restrict__ out,
                                        const T_IN* __restrict__ input,
                                        const float* __restrict__ input_scale_inv,
                                        int64_t num_elems) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  // Invert the scale so that we can use multiplications to avoid expensive
  // division.

  // Inverse was completed when the weight was loaded!
  //const float inverted_scale = 1.0f / (*scale);
  scaled_fp8_conversion_vec<T_IN, true>(
      out, input, (*input_scale_inv), num_elems, tid, blockDim.x * gridDim.x);
}

template<typename T_IN, QUANTIZE_MODE quantize_mode>
void invokeScaleFP8QuantMatrix(
    __nv_fp8_e4m3* output, float const* input_scale_inv, T_IN const* input, uint32_t size, uint32_t n, cudaStream_t stream)
{
    if(quantize_mode !=  QUANTIZE_MODE::PER_TENSOR)
    {
        TM_LOG_INFO("Only Support QUANTIZE_MODE::PER_TENSO Mode!");
        return;
    }

    dim3 grid(size);
    dim3 block(1024);
    scaled_fp8_quant_kernel<T_IN><<<grid, block, 0, stream>>>(
        output, input, input_scale_inv, n);
}

#define defineinvokeScaleFP8QuantMatrix(type_in, mode)                                                                 \
    template void invokeScaleFP8QuantMatrix<type_in, mode>(__nv_fp8_e4m3 * output,                                     \
                                                           float const*   input_scale,                                 \
                                                           type_in const* input,                                       \
                                                           uint32_t       size,                                        \
                                                           uint32_t       n,                                           \
                                                           cudaStream_t   stream);

defineinvokeScaleFP8QuantMatrix(half, QUANTIZE_MODE::PER_TENSOR);
#ifdef ENABLE_BF16
defineinvokeScaleFP8QuantMatrix(__nv_bfloat16, QUANTIZE_MODE::PER_TENSOR);
#endif

template<typename T_OUT, typename T_IN, typename T_FAKE>
__global__ void fakeQuantize(T_OUT* dst, const T_IN* src, const int size)
{
    for (int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid += blockDim.x * gridDim.x) {
        T_FAKE tmp = (T_FAKE)((float)src[tid]);
        dst[tid]   = (T_OUT)((float)tmp);
    }
}

template<typename T_OUT, typename T_IN, typename T_FAKE>
void invokeFakeQuantize(T_OUT* dst, const T_IN* src, const int size, cudaStream_t stream)
{
    fakeQuantize<T_OUT, T_IN, T_FAKE><<<256, 256, 0, stream>>>(dst, src, size);
}

template void
invokeFakeQuantize<float, float, __nv_fp8_e4m3>(float* dst, const float* src, const int size, cudaStream_t stream);
template void
invokeFakeQuantize<half, half, __nv_fp8_e4m3>(half* dst, const half* src, const int size, cudaStream_t stream);
template void invokeFakeQuantize<__nv_bfloat16, __nv_bfloat16, __nv_fp8_e4m3>(__nv_bfloat16*       dst,
                                                                              const __nv_bfloat16* src,
                                                                              const int            size,
                                                                              cudaStream_t         stream);

template<typename T_W>
__global__ void computeFP8QuantizeScale(float* quant_ptr, const T_W* weights, const int k, const int n)
{
    float max = -10000.f;
    for (int i = 0; i < k; i++) {
        float val = fabs((float)weights[i * n + blockIdx.x * blockDim.x + threadIdx.x]);
        max       = max > val ? max : val;
        // if (threadIdx.x == 0 && blockIdx.x == 0 && i % 100 == 0) {
        //     printf("max: %f, val: %f \n", max, val);
        // }
    }
    // quant_ptr[blockIdx.x * blockDim.x + threadIdx.x] = 1.0f;
    // quant_ptr[blockIdx.x * blockDim.x + threadIdx.x] = FP8_E4M3_MAX / max;
    quant_ptr[blockIdx.x * blockDim.x + threadIdx.x] = std::max(max / FP8_E4M3_MAX, 1.0f / 32.f);
}

template<typename T_W>
void invokeComputeFP8QuantizeScale(float* quant_ptr, const T_W* weights, const int k, const int n, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid;
    grid.x = (n + 255) / 256;
    computeFP8QuantizeScale<T_W><<<grid, block, 0, stream>>>(quant_ptr, weights, k, n);
}

#ifdef ENABLE_BF16
template void invokeComputeFP8QuantizeScale(
    float* quant_ptr, const __nv_bfloat16* weights, const int k, const int n, cudaStream_t stream);
#endif
template void
invokeComputeFP8QuantizeScale(float* quant_ptr, const float* weights, const int k, const int n, cudaStream_t stream);

#endif  // ENABLE_FP8
}  // namespace turbomind
