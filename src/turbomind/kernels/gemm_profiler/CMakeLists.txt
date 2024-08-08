# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(gemm_profiler)

if(FUSED_GATED_GEMM)
    list(APPEND FUSED_GATED_SRC
        gemmSwigluProfiler.cpp
        gemmSwigluProfiler.cu)
endif()

if(FUSED_MOE_FFN_GEMM)
    list(APPEND FUSED_MOE_SRC
        gemmMoEProfiler.cpp
        gemmMoEProfiler.cu)
endif()

add_library(${PROJECT_NAME} STATIC
        gemmPluginProfiler.cpp
        ${FUSED_GATED_SRC}
        ${FUSED_MOE_SRC})

target_include_directories(${PROJECT_NAME} PRIVATE ${CUTLASS_HEADER_DIR} ${CUTLASS_TOOLS_HEAD_DIR} ${CUTLASS_EXTENSIONS_DIR})
target_compile_options(${PROJECT_NAME} PRIVATE -O3)
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
