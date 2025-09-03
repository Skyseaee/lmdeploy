// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/utils/rng_utils.h"
#include "src/turbomind/macro.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

#include "src/turbomind/core/core.h"

namespace turbomind {

template<typename T>
void Compare(const T* src,
             const T* ref,
             size_t   stride,
             int      dims,
             int      bsz,
             bool     show = false,
             float    rtol = 1e-2,
             float    atol = 1e-4);

void Compare(const void* x,
             const void* r,
             DataType    dtype,
             size_t      stride,
             int         dim,
             int         bsz,
             bool        show,
             float       rtol = 1e-2,
             float       atol = 1e-4);

template<class T>
std::vector<float> FastCompare(const T*     src,  //
                               const T*     ref,
                               int          dims,
                               int          bsz,
                               cudaStream_t stream,
                               float        rtol = 1e-2,
                               float        atol = 1e-4);

std::vector<float> FastCompare(const Tensor& x,  //
                               const Tensor& r,
                               cudaStream_t  stream,
                               float         rtol = 1e-2,
                               float         atol = 1e-4);

void FC_Header();

void FC_Print(const std::vector<float>& d);

void LoadBinary(const std::string& path, size_t size, void* dst);

}  // namespace turbomind
