// Copyright (c) OpenMMLab. All rights reserved.

#include "test_utils.h"
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <fstream>
#include <iostream>

#define _CG_ABI_EXPERIMENTAL
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

namespace turbomind {

cublasHandle_t cublas_handle{};
cudaStream_t   cublas_stream{};

template<typename T>
void Compare(const T* src, const T* ref, size_t stride, int m, int n, bool show, float rtol, float atol)
{
    float asums{};
    float rsums{};
    int   outliers{};
    for (int nn = 0; nn < n; ++nn) {
        float abs_diff_sum{};
        float rel_diff_sum{};
        for (int mm = 0; mm < m; ++mm) {
            auto x = float(src[nn * stride + mm]);
            auto y = float(ref[nn * stride + mm]);
            // if (show) {
            //     std::cout << x << "\t" << y << std::endl;
            // }
            auto abs_diff = std::abs(x - y);
            auto rel_diff = abs_diff / std::abs(y + 1e-6f);
            if (!(abs_diff <= atol + rtol * std::abs(y))) {
                ++outliers;
                if (show) {
                    std::cout << nn << "," << mm << "\t" << x << "\t" << y << std::endl;
                }
            }
            abs_diff_sum += abs_diff;
            rel_diff_sum += rel_diff;
        }
        asums += abs_diff_sum / m;
        rsums += rel_diff_sum / m;
    }
    std::cout << "abs_diff = " << asums / n << " rel_diff = " << rsums / n << " outliers = " << outliers / (float)n
              << std::endl;
}

template void Compare(const half* src, const half* ref, size_t stride, int m, int n, bool show, float rtol, float atol);
template void
Compare(const float* src, const float* ref, size_t stride, int m, int n, bool show, float rtol, float atol);
#if ENABLE_BF16
template void
Compare(const nv_bfloat16* src, const nv_bfloat16* ref, size_t stride, int m, int n, bool show, float rtol, float atol);
#endif
#if ENABLE_FP8
template void
Compare(const __nv_fp8_e4m3* src, const __nv_fp8_e4m3* ref, size_t stride, int m, int n, bool show, float rtol, float atol);
#endif

void LoadBinary(const std::string& path, size_t size, void* dst)
{
    std::ifstream ifs(path, std::ios::binary | std::ios::in);
    if (!ifs.is_open()) {
        std::cerr << "failed to open " << path << "\n";
        std::abort();
    }
    ifs.seekg(0, ifs.end);
    auto actual_size_in_bytes = ifs.tellg();
    ifs.seekg(0, ifs.beg);
    if (size != actual_size_in_bytes) {
        std::cerr << "[warning] file " << path << " has " << actual_size_in_bytes << " bytes, while " << size
                  << " bytes is requested\n";
    }
    ifs.read((char*)dst, size);
    std::cerr << "[info] " << path << " " << size << "\n";
}

}  // namespace turbomind
