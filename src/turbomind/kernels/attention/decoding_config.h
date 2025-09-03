// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "arch.h"
#include "block_iterator.h"
#include "cta_map.h"
#include "impl_81616.h"
#include "impl_81616_fp8.h"
#include "impl_81616_fp8_dynamic.h"
#include "impl_simt.h"
#include "mainloop_sm70.h"
#include "mainloop_sm80.h"
#include "mainloop_sm80_fp8.h"
#include "mainloop_sm80_fp8_dynamic.h"
#include "src/turbomind/kernels/attention/attention_universal.h"
#include "src/turbomind/kernels/attention/impl.h"
#include "src/turbomind/kernels/attention/mainloop.h"

namespace turbomind::attention {

template<class Arch, class T, class Tkv, int Qh, int HeadDim, class SFINAE = void>
struct DecodingConfig {
    static_assert(sizeof(T) == 0, "config not found");
};

template<class Arch, class T, class Tkv, int Qh, int HeadDim>
using Decoding = typename DecodingConfig<Arch, T, Tkv, Qh, HeadDim>::Kernel;

struct Base_1x64_1x16 {
    static constexpr int CTA_Q  = 1;
    static constexpr int CTA_S  = 64;
    static constexpr int WARP_Q = 1;
    static constexpr int WARP_S = 16;
};

//////////////////////////////////////////////////////////////
template<class T, int Qh, int HeadDim>
struct DecodingConfig<arch::Sm80, T, T, Qh, HeadDim, std::enable_if_t<!(Qh > 2)>> {
    using Attention = Impl<MMA_SIMT, T, T, Qh, 1, 64, Qh, 1, 16, HeadDim, 3>;
    using CacheIter = GetBlockIterFactory<T, T, 64, HeadDim, false>;
    using Kernel    = AttentionUniversal<arch::Sm80, Mainloop<Sm80_CpAsync<3>, Attention>, CacheIter, DecodingCtaMap, false>;
};

template<class T, int Qh_, int HeadDim>
struct DecodingConfig<arch::Sm80, T, T, Qh_, HeadDim, std::enable_if_t<(Qh_ > 2)>> {
    static constexpr int Qh = (Qh_ + 7) / 8 * 8;
    using Attention         = Impl<MMA_81616, T, T, Qh, 1, 64, Qh, 1, 16, HeadDim, 3>;
    using CacheIter         = GetBlockIterFactory<T, T, 64, HeadDim, false>;
    using Kernel = AttentionUniversal<arch::Sm80, Mainloop<Sm80_CpAsync<3>, Attention>, CacheIter, DecodingCtaMap, false>;
};

template<class T, int Qh_, int HeadDim>
struct DecodingConfig<arch::Sm80, T, uint8_t, Qh_, HeadDim, std::enable_if_t<(HeadDim != 192)>> {
    static constexpr int Qh = (Qh_ + 7) / 8 * 8;
    using Attention         = Impl<MMA_81616, T, uint8_t, Qh, 1, 64, Qh, 1, 16, HeadDim, 5>;
    using CacheIter         = GetBlockIterFactory<T, uint8_t, 64, HeadDim, false>;
    using Kernel = AttentionUniversal<arch::Sm80, Mainloop<Sm80_CpAsync<5>, Attention>, CacheIter, DecodingCtaMap, false>;
};

template<class T, int Qh_, int HeadDim>
struct DecodingConfig<arch::Sm80, T, uint4_t, Qh_, HeadDim> {
    static constexpr int Qh = (Qh_ + 7) / 8 * 8;
    using Attention         = Impl<MMA_81616, T, uint4_t, Qh, 1, 64, Qh, 1, 16, HeadDim, 5>;
    using CacheIter         = GetBlockIterFactory<T, uint4_t, 64, HeadDim, false>;
    using Kernel = AttentionUniversal<arch::Sm80, Mainloop<Sm80_CpAsync<5>, Attention>, CacheIter, DecodingCtaMap, false>;
};

// NOTE(Alan): for fp8 kv quant static
template<class T, int Qh_, int HeadDim>
struct DecodingConfig<arch::Sm80, T, __nv_fp8_e4m3, Qh_, HeadDim> {
    static constexpr int Qh = (Qh_ + 7) / 8 * 8;
    using Attention         = Impl<MMA_81616_FP8, T, __nv_fp8_e4m3, Qh, 1, 64, Qh, 1, 16, HeadDim, 5>;
    using CacheIter         = GetBlockIterFactory<T, __nv_fp8_e4m3, 64, HeadDim, true>;
    using Kernel = AttentionUniversal<arch::Sm80, MainloopFP8<Sm80_CpAsync<5>, Attention>, CacheIter, DecodingCtaMap, true>;
};

// NOTE(Alan): for fp8 kv quant dynamic with scale
template<class T, int Qh_, int HeadDim>
struct DecodingConfig<arch::Sm80, T, fp8_dynamic, Qh_, HeadDim> {
    static constexpr int Qh = (Qh_ + 7) / 8 * 8;
    using Attention         = Impl<MMA_81616_FP8_Dynamic, T, uint8_t, Qh, 1, 64, Qh, 1, 16, HeadDim, 5>;
    using CacheIter         = GetBlockIterFactory<T, uint8_t, 64, HeadDim, false>;
    using Kernel = AttentionUniversal<arch::Sm80, MainloopFP8Dynamic<Sm80_CpAsync<5>, Attention>, CacheIter, DecodingCtaMap, false>;
};
//////////////////////////////////////////////////////////////

template<class T, int Qh_, int HeadDim>
struct DecodingConfig<arch::Sm75, T, T, Qh_, HeadDim> {
    static constexpr int Qh = (Qh_ + 7) / 8 * 8;
    using Attention         = Impl<MMA_81616, T, T, Qh, 1, 64, Qh, 1, 16, HeadDim, 2>;
    using CacheIter         = GetBlockIterFactory<T, T, 64, HeadDim, false>;
    using Kernel = AttentionUniversal<arch::Sm75, Mainloop<arch::Sm70, Attention>, CacheIter, DecodingCtaMap, false>;
};

template<class T, int Qh_, int HeadDim>
struct DecodingConfig<arch::Sm75, T, uint8_t, Qh_, HeadDim> {
    static constexpr int Qh = (Qh_ + 7) / 8 * 8;
    using Attention         = Impl<MMA_81616, T, uint8_t, Qh, 1, 64, Qh, 1, 16, HeadDim, 2>;
    using CacheIter         = GetBlockIterFactory<T, uint8_t, 64, HeadDim, false>;
    using Kernel = AttentionUniversal<arch::Sm75, Mainloop<arch::Sm70, Attention>, CacheIter, DecodingCtaMap, false>;
};

template<class T, int Qh_, int HeadDim>
struct DecodingConfig<arch::Sm75, T, uint4_t, Qh_, HeadDim> {
    static constexpr int Qh = (Qh_ + 7) / 8 * 8;
    using Attention         = Impl<MMA_81616, T, uint4_t, Qh, 1, 64, Qh, 1, 16, HeadDim, 2>;
    using CacheIter         = GetBlockIterFactory<T, uint4_t, 64, HeadDim, false>;
    using Kernel = AttentionUniversal<arch::Sm75, Mainloop<arch::Sm70, Attention>, CacheIter, DecodingCtaMap, false>;
};

//////////////////////////////////////////////////////////////

template<class T, int Qh, int HeadDim>
struct DecodingConfig<arch::Sm70, T, T, Qh, HeadDim> {
    // Qh >= 4 is not beneficial for sm_70
    static constexpr int kH = Qh % 3 == 0 ? 3 : (Qh % 2 == 0 ? 2 : 1);
    using Attention         = Impl<MMA_SIMT, T, T, kH, 1, 64, kH, 1, 16, HeadDim, 2>;
    using CacheIter         = GetBlockIterFactory<T, T, 64, HeadDim, false>;
    using Kernel = AttentionUniversal<arch::Sm70, Mainloop<arch::Sm70, Attention>, CacheIter, DecodingCtaMap, false>;
};

template<class T, int Qh, int HeadDim>
struct DecodingConfig<arch::Sm70, T, uint8_t, Qh, HeadDim> {
    // Qh >= 4 is not beneficial for sm_70
    static constexpr int kH = Qh % 3 == 0 ? 3 : (Qh % 2 == 0 ? 2 : 1);
    using Attention         = Impl<MMA_SIMT, T, uint8_t, kH, 1, 64, kH, 1, 16, HeadDim, 2>;
    using CacheIter         = GetBlockIterFactory<T, uint8_t, 64, HeadDim, false>;
    using Kernel = AttentionUniversal<arch::Sm70, Mainloop<arch::Sm70, Attention>, CacheIter, DecodingCtaMap, false>;
};

template<class T, int Qh, int HeadDim>
struct DecodingConfig<arch::Sm70, T, uint4_t, Qh, HeadDim> {
    // Qh >= 4 is not beneficial for sm_70
    static constexpr int kH = Qh % 3 == 0 ? 3 : (Qh % 2 == 0 ? 2 : 1);
    using Attention         = Impl<MMA_SIMT, T, uint4_t, kH, 1, 64, kH, 1, 16, HeadDim, 2>;
    using CacheIter         = GetBlockIterFactory<T, uint4_t, 64, HeadDim, false>;
    using Kernel = AttentionUniversal<arch::Sm70, Mainloop<arch::Sm70, Attention>, CacheIter, DecodingCtaMap, false>;
};

template<class T>
struct DecodingConfig<arch::Sm80, T, uint8_t, 1, 192> {
    static constexpr int Qh      = 1;
    static constexpr int HeadDim = 192;

    using Attention = Impl<MMA_SIMT, T, uint8_t, Qh, 1, 64, Qh, 1, 16, HeadDim, 3>;
    using CacheIter = GetBlockIterFactory<T, uint8_t, 64, HeadDim, false>;
    using Kernel =
        AttentionUniversal<arch::Sm80, Mainloop<Sm80_CpAsync<3>, Attention>, CacheIter, DecodingCtaMap, false>;
};

}  // namespace turbomind::attention
