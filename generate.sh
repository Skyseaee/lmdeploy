#!/bin/bash
WORKSPACE_PATH=$(dirname "$(readlink -f "$0")")

builder="-G Ninja"

if [ "$1" == "make" ]; then
    builder=""
fi

SM="90"

cmake ${builder} .. \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DCMAKE_INSTALL_PREFIX=${WORKSPACE_PATH}/install \
    -DBUILD_PY_FFI=ON \
    -DBUILD_MULTI_GPU=ON \
    -DCMAKE_CUDA_FLAGS="-lineinfo" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DUSE_NVTX=ON \
    -DENABLE_FP8=ON \
    -DCUTLASS_FP8=ON \
    -DFUSED_GATED_GEMM=ON \
    -DFUSED_MOE_FFN_GEMM=ON \
    -DBUILD_TEST=OFF # -DCMAKE_CUDA_ARCHITECTURES=${SM}
