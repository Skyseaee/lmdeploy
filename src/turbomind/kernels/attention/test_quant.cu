// Copyright (c) OpenMMLab. All rights reserved.

#include "quantization.h"
#include "src/turbomind/kernels/attention/test_utils.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"
#include "src/turbomind/macro.h"
#include <cstdint>
#include <iostream>
#include <thrust/universal_vector.h>

using namespace turbomind;

template<int kVecSize, class T0, class T1>
__global__ void convert(T1* dst, const T0* src, size_t n, float scale, float zero, bool static_flag = false)
{
    auto v_src = (Array<T0, kVecSize>*)src;
    auto v_dst = (Array<T1, kVecSize>*)dst;

    const int v_n = n / kVecSize;

    ConvertKvCache<T0, T1> converter{scale, zero};

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < v_n; i += blockDim.x * gridDim.x) {
        Array<T0, kVecSize> vi;
        Array<T1, kVecSize> vo;
        Load(vi, (T0*)v_src[i].data());

        if (static_flag) {
            if constexpr (std::is_same_v<T0, __nv_fp8_e4m3> || std::is_same_v<T1, __nv_fp8_e4m3>)
            {
                vo = ConvertKvCache<T0, T1>::convert(vi);
            }
        } else {
            vo = converter(vi);
        }
        Store((T1*)v_dst[i].data(), vo);
    }
}

template<class T0, class T1, int kVecSize>
void round_trip_test(size_t n, float s1 = 1., float z1 = 0., float s2 = 1., float z2 = 0., bool static_flag = false)
{
    std::cout << __PRETTY_FUNCTION__ << std::endl;

    using namespace thrust;

    universal_vector<T0> src(n);
    universal_vector<T0> dst(src.size());

    universal_vector<Array<T1, kVecSize>> tmp(src.size() / kVecSize);

    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = T0(float(rand() % (1 << bitsof<T1>)));
        // printf("%.f\n", float(src[i]));
    }

    convert<kVecSize><<<256, 256>>>((T1*)tmp.data().get(), src.data().get(), n, s1, z1);
    convert<kVecSize><<<256, 256>>>(dst.data().get(), (const T1*)tmp.data().get(), n, s2, z2);

    cudaDeviceSynchronize();

    Compare(dst.data().get(), src.data().get(), src.size(), src.size(), 1);
}

int main(int argc, char* argv[])
{
    round_trip_test<float, uint8_t, 4>(1 << 20);
    round_trip_test<half, uint8_t, 4>(1 << 20);
#if ENABLE_BF16
    round_trip_test<nv_bfloat16, uint8_t, 4>(1 << 20);
#endif

    round_trip_test<float, uint4_t, 8>(1 << 20, 1, 0, 1, -64);
    round_trip_test<half, uint4_t, 8>(1 << 20, 1, 0, 1, -64);
#if ENABLE_BF16
    round_trip_test<nv_bfloat16, uint4_t, 8>(1 << 20, 1, 0, 1, 0);
#endif

#if ENABLE_FP8
    round_trip_test<__nv_fp8_e4m3, half, 4>(1 << 20);
    round_trip_test<half, __nv_fp8_e4m3, 4>(1 << 20);

    round_trip_test<__nv_fp8_e4m3, nv_bfloat16, 4>(1 << 20);
    round_trip_test<nv_bfloat16, __nv_fp8_e4m3, 4>(1 << 20);


    round_trip_test<__nv_fp8_e4m3, half, 4>(1 << 20, 1, 0, 1, 0, true);
    round_trip_test<half, __nv_fp8_e4m3, 4>(1 << 20, 1, 0, 1, 0, true);

    round_trip_test<__nv_fp8_e4m3, nv_bfloat16, 4>(1 << 20, 1, 0, 1, 0, true);
    round_trip_test<nv_bfloat16, __nv_fp8_e4m3, 4>(1 << 20, 1, 0, 1, 0, true);
#endif

    return 0;
}
