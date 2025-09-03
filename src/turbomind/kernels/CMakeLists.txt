# Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.8)

add_library(ban_bad_words STATIC ban_bad_words.cu)
set_property(TARGET ban_bad_words PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ban_bad_words PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(stop_criteria STATIC stop_criteria_kernels.cu)
set_property(TARGET stop_criteria PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET stop_criteria PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(activation_kernels STATIC activation_kernels.cu)
set_property(TARGET activation_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET activation_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(quantization_kernels STATIC quantization.cu)
set_property(TARGET quantization_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET quantization_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_executable(test_quantization test_quantization.cc gemm/test/test_utils.cu)
target_link_libraries(test_quantization PRIVATE quantization_kernels core rng_utils)

add_library(logprob_kernels STATIC logprob_kernels.cu)
set_property(TARGET logprob_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET logprob_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(unfused_attention_kernels STATIC unfused_attention_kernels.cu)
set_property(TARGET unfused_attention_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET unfused_attention_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(decoding_kernels STATIC decoding_kernels.cu)
set_property(TARGET decoding_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET decoding_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(gpt_kernels STATIC gpt_kernels.cu)
set_property(TARGET gpt_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET gpt_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(sampling_topk_kernels STATIC sampling_topk_kernels.cu)
set_property(TARGET sampling_topk_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET sampling_topk_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(sampling_topp_kernels STATIC sampling_topp_kernels.cu)
set_property(TARGET sampling_topp_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET sampling_topp_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(sampling_penalty_kernels STATIC sampling_penalty_kernels.cu)
set_property(TARGET sampling_penalty_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET sampling_penalty_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(custom_ar_kernels STATIC custom_ar_kernels.cu)
set_property(TARGET custom_ar_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET custom_ar_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(sampling_kernels STATIC sampling_kernels.cu)
set_property(TARGET sampling_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET sampling_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(pre_quantscale_kernel STATIC pre_quantscale_kernel.cu)
set_property(TARGET pre_quantscale_kernel PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET pre_quantscale_kernel PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(moe_load_balance_kernels STATIC moe_load_balance_kernels.cu)
set_property(TARGET moe_load_balance_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET moe_load_balance_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)


# if (BUILD_TEST)
#     add_subdirectory(flash_attention)
# endif ()

add_subdirectory(attention)
add_subdirectory(gemm)
add_subdirectory(norm)

if(CUTLASS_FP8)
    add_subdirectory(cutlass_w8a8)
endif()

if(FUSED_GATED_GEMM OR FUSED_MOE_FFN_GEMM OR CUTLASS_FP8)
    add_subdirectory(gemm_profiler)
endif()

if(FUSED_GATED_GEMM)
    add_subdirectory(fused_gated_gemm)
endif()

if(FUSED_MOE_FFN_GEMM)
    add_subdirectory(lora)
    add_subdirectory(cutlass_kernels)
endif()