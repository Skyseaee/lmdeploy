# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.8)
project(cutlass_moe_gemm)

# The Python executable will only be defined if building with Torch support. If
# not, we need to find it here.
if(NOT Python3_EXECUTABLE)
  find_package(
    Python3
    COMPONENTS Interpreter
    REQUIRED)
endif()

execute_process(
  WORKING_DIRECTORY ${CUTLASS_DIR}/python/
  COMMAND ${Python3_EXECUTABLE} setup_library.py develop
  RESULT_VARIABLE _CUTLASS_LIBRARY_SUCCESS)

if(NOT _CUTLASS_LIBRARY_SUCCESS MATCHES 0)
  message(
    FATAL_ERROR
      "Failed to set up the CUTLASS library due to ${_CUTLASS_LIBRARY_SUCCESS}."
  )
endif()

set_directory_properties(
  PROPERTIES CMAKE_CONFIGURE_DEPENDS
             ${CMAKE_CURRENT_SOURCE_DIR}/python/generate_kernels.py)

set(INSTANTIATION_GENERATION_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/cutlass_instantiations)
execute_process(
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/
  COMMAND ${Python3_EXECUTABLE} generate_kernels.py -o
          ${INSTANTIATION_GENERATION_DIR}
  RESULT_VARIABLE _KERNEL_GEN_SUCCESS)

if(NOT _KERNEL_GEN_SUCCESS MATCHES 0)
  message(
    FATAL_ERROR
      "Failed to generate CUTLASS kernel instantiations due to ${_KERNEL_GEN_SUCCESS}."
  )
endif()

# Get the sources for MOE Grouped GEMM launchers
file(GLOB_RECURSE GROUPED_CU_INSTANTIATIONS
     ${INSTANTIATION_GENERATION_DIR}/gemm_grouped/*.cu)
file(GLOB_RECURSE GROUPED_SRC_CPP ./*.cpp)
file(GLOB_RECURSE GROUPED_SRC_CU ./*.cu)

message(
  VERBOSE
  "Group srcs ${GROUPED_SRC_CU} ${GROUPED_SRC_CPP} ${GROUPED_CU_INSTANTIATIONS}"
)

# WARNING: Building with `-G` flag may generate invalid results for this target
add_library(cutlass_moe_gemm STATIC ${GROUPED_SRC_CU} ${GROUPED_SRC_CPP}
                                ${GROUPED_CU_INSTANTIATIONS})

foreach(target_name
          cutlass_moe_gemm;)
  set_property(TARGET ${target_name} PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET ${target_name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  # Note - we deliberately do not include 90a PTX (even when 9.0+PTX is
  # specified). This is because sm_90a has arch conditional instructions that
  # are not forward compatible. As a result, it does not make sense to embed PTX
  # into the binary anyway.
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 12.0)

    message(STATUS "MANUALLY APPENDING FLAG TO COMPILE FOR SM_90a.")
    target_compile_options(
      ${target_name}
      PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_90a,code=sm_90a>)

    target_include_directories(${target_name} PRIVATE ${CUTLASS_HEADER_DIR} ${CUTLASS_TOOLS_HEAD_DIR} ${CUTLASS_EXTENSIONS_DIR})

    target_link_libraries(${target_name} PRIVATE nvidia::cutlass::cutlass fused_gated_gemm)

    # Hopper kernels require cuda lib for TMA APIs
    # target_link_libraries(${target_name} PRIVATE CUDA::cuda_driver)

    # No kernels should be parsed, unless hopper is specified. This is a build
    # time improvement
    target_compile_definitions(${target_name} PRIVATE COMPILE_HOPPER_TMA_GEMMS)
  endif()

  # Suppress GCC note: the ABI for passing parameters with 64-byte alignment has
  # changed in GCC 4.6 This note appears for kernels using TMA and clutters the
  # compilation output.
  if(NOT WIN32)
    target_compile_options(
      ${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-psabi>)
  endif()

endforeach()
