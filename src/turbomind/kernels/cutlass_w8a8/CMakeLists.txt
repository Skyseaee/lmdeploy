# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.8)
project(cutlass_w8a8)

add_library(${PROJECT_NAME} STATIC scaled_mm_entry.cu scaled_mm_c3x.cu scaled_mm_c2x.cu)

# The CUTLASS kernels for Hopper require sm90a to be enabled.
# This is done via the below gencode option, BUT that creates kernels for both sm90 and sm90a.
# That adds an extra 17MB to compiled binary, so instead we selectively enable it.
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 12.0)
  set_source_files_properties(
    "scaled_mm_c3x.cu"
    PROPERTIES
    COMPILE_FLAGS
    "-gencode arch=compute_90a,code=sm_90a")
else()
  message(FATAL_ERROR "cutlass fp8 gemm kernel required sm 90a")
endif()

#
# For the cutlass_scaled_mm kernels we want to build the c2x (CUTLASS 2.x)
# kernels for the remaining archs that are not already built for 3x.
# (Build 8.9 for FP8)
cuda_archs_loose_intersection(SCALED_MM_2X_ARCHS
  "8.9+PTX" "${CMAKE_CUDA_ARCHITECTURES}")
# subtract out the archs that are already built for 3x
# list(REMOVE_ITEM SCALED_MM_2X_ARCHS ${SCALED_MM_3X_ARCHS})
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 11.8)
  if (SCALED_MM_2X_ARCHS)
    add_definitions(-DENABLE_SCALED_MM_C2X=1)
    set_source_files_properties(
      "scaled_mm_c2x.cu"
      PROPERTIES
      COMPILE_FLAGS
        "-gencode arch=compute_89,code=sm_89 \
        -gencode arch=compute_89,code=compute_89")
      message(STATUS "Building scaled_mm_c2x for archs: ${SCALED_MM_2X_ARCHS}")
  else()
    # if (SCALED_MM_3X_ARCHS)
    #   message(STATUS "Not building scaled_mm_c2x as all archs are already built"
    #                  " for and covered by scaled_mm_c3x")
    # else()
    message(STATUS "Not building scaled_mm_c2x as no compatible archs found "
                "in CUDA target architectures")
    #endif()
  endif()
else()
  message(STATUS "The Ada Lovelace architecture has been supported since CUDA 11.8. ")
endif()

target_include_directories(${PROJECT_NAME} PRIVATE ${CUTLASS_HEADER_DIR} ${CUTLASS_TOOLS_HEAD_DIR} ${CUTLASS_EXTENSIONS_DIR})

target_link_libraries(${PROJECT_NAME} PRIVATE nvidia::cutlass::cutlass)
target_compile_options(${PROJECT_NAME} PRIVATE
  --generate-line-info -O3 -use_fast_math -Xptxas=-v --expt-relaxed-constexpr)
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
