# Copyright (c) OpenMMLab. All rights reserved.
cmake_minimum_required(VERSION 3.8)
project(lora)

add_library(${PROJECT_NAME} STATIC lora.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${CUTLASS_HEADER_DIR} ${CUTLASS_TOOLS_HEAD_DIR} ${CUTLASS_EXTENSIONS_DIR})

target_link_libraries(${PROJECT_NAME} PRIVATE nvidia::cutlass::cutlass)

target_compile_options(${PROJECT_NAME} PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:
		-O3
		-use_fast_math
                -Xptxas=-v
                --generate-line-info
                --threads 16>
)

set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

