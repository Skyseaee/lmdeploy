# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# GoogleTest Preparation - Code block copied from
# https://google.github.io/googletest/quickstart-cmake.html
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)

find_package(CUDAToolkit REQUIRED)

if(NOT MSVC)
  add_definitions(-DTORCH_CUDA=1)
endif()

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include_directories(
  ${PROJECT_SOURCE_DIR}
)

remove_definitions(-DTORCH_CUDA=1)
add_executable(test_fuse_gated_gemm_kernel gemmSwigluKernelTestSm90Fp8.cu)
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 12.0)
  set_source_files_properties(
    "gemmSwigluKernelTestSm90Fp8.cu"
    PROPERTIES
    COMPILE_FLAGS
    "-gencode arch=compute_90a,code=sm_90a")
else()
  message(FATAL_ERROR "cutlass fused_gated_gemm kernel required sm 90a")
endif()

target_link_libraries(test_fuse_gated_gemm_kernel PUBLIC CUDA::cublas CUDA::cudart CUDA::curand gemm cublasMMWrapper cublasFP8MMWrapper fused_gated_gemm tensor cutlass_w8a8 cuda_fp8_utils cuda_utils logger)
target_include_directories(test_fuse_gated_gemm_kernel PRIVATE ${CUTLASS_HEADER_DIR} ${CUTLASS_TOOLS_HEAD_DIR} ${CUTLASS_EXTENSIONS_DIR})
target_link_libraries(test_fuse_gated_gemm_kernel PRIVATE nvidia::cutlass::cutlass)

add_executable(test_fuse_gated_gemm_runner gemmSwigluRunnerTest.cu)
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 12.0)
  set_source_files_properties(
    "gemmSwigluRunnerTest.cu"
    PROPERTIES
    COMPILE_FLAGS
    "-gencode arch=compute_90a,code=sm_90a")
else()
  message(FATAL_ERROR "cutlass fused_gated_gemm kernel required sm 90a")
endif()
target_link_libraries(test_fuse_gated_gemm_runner PUBLIC CUDA::cublas CUDA::cudart CUDA::curand gemm cublasMMWrapper cublasFP8MMWrapper fused_gated_gemm tensor cutlass_w8a8 cuda_fp8_utils cuda_utils logger)
target_include_directories(test_fuse_gated_gemm_runner PRIVATE ${CUTLASS_HEADER_DIR} ${CUTLASS_TOOLS_HEAD_DIR} ${CUTLASS_EXTENSIONS_DIR})
target_link_libraries(test_fuse_gated_gemm_runner PRIVATE nvidia::cutlass::cutlass)