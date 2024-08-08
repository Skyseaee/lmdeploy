# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.8)
project(mixture_of_experts)

add_library(${PROJECT_NAME} STATIC moe_kernels.cu)

# The CUTLASS kernels for Hopper require sm90a to be enabled.
# This is done via the below gencode option, BUT that creates kernels for both sm90 and sm90a.
# That adds an extra 17MB to compiled binary, so instead we selectively enable it.
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 12.0)
  set_source_files_properties(
    "moe_kernels.cu"
    PROPERTIES
    COMPILE_FLAGS
    "-gencode arch=compute_90a,code=sm_90a")
else()
  message(FATAL_ERROR "moe_kernels gemm kernel required sm 90a")
endif()

target_include_directories(${PROJECT_NAME} PRIVATE ${CUTLASS_HEADER_DIR} ${CUTLASS_TOOLS_HEAD_DIR} ${CUTLASS_EXTENSIONS_DIR})

target_link_libraries(${PROJECT_NAME} PRIVATE cutlass_moe_gemm lora)
target_link_libraries(${PROJECT_NAME} PRIVATE nvidia::cutlass::cutlass)
target_compile_options(${PROJECT_NAME} PRIVATE
  --generate-line-info -O3 -use_fast_math -Xptxas=-v --expt-relaxed-constexpr)
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_subdirectory(moe_gemm)
