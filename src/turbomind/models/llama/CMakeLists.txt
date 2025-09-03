# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.8)


find_package(CUDAToolkit REQUIRED)

add_library(Llama STATIC
        LlamaV2.cc
        LlamaBatch.cc
        LlamaLinear.cu
        BlockManager.cc
        BlockTrie.cc
        SequenceManager.cc
        LlamaWeight.cc
        LlamaDenseWeight.cc
        LlamaDecoderLayerWeight.cc
        LlamaFfnLayer.cc
        moe_ffn_layer.cc
        unified_decoder.cc
        unified_attention_layer.cc
        llama_kernels.cu
        llama_utils.cu
        mla_utils.cu)
set_property(TARGET Llama PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Llama PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
set(LLAMA_LIBRARIES CUDA::cudart
        engine
        core
        gemm2
        CUDA::cublas
        rms_norm
        DynamicDecodeLayer
        activation_kernels
        attention
        decoding_kernels
        quantization_kernels
        unfused_attention_kernels
        gpt_kernels
        memory_utils
        cuda_utils
        logger
        anomaly_handler
        rng_utils
        rms_norm
        cublasMMWrapper
        custom_ar_kernels
        pre_quantscale_kernel
        env_utils)

if(ENABLE_FP8)
        list(APPEND LLAMA_LIBRARIES cuda_fp8_utils cublasFP8MMWrapper)
endif()

if(CUTLASS_FP8)
        list(APPEND LLAMA_LIBRARIES cutlass_w8a8)
endif()

if(FUSED_GATED_GEMM)
        list(APPEND LLAMA_LIBRARIES fused_gated_gemm gemm_profiler cutlass_src)
endif()

if(FUSED_MOE_FFN_GEMM)
        list(APPEND LLAMA_LIBRARIES gemm_profiler moe_gemm_src cutlass_src fp8_blockscale_gemm_src moe_load_balance_kernels fpA_intB_gemm_src) # mixture_of_experts
endif()

target_link_libraries(Llama PUBLIC ${LLAMA_LIBRARIES})
