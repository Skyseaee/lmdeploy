// Copyright (c) OpenMMLab. All rights reserved.

#include "../decoding_config.h"
#include "../decoding_template.h"

namespace turbomind {

using namespace attention;

#ifdef ENABLE_FP8
template bool invokeDecoding<Decoding<arch::Sm80, half, __nv_fp8_e4m3, 8, 64>>(const AttentionParams<half>&);

template bool invokeDecoding<Decoding<arch::Sm80, half, __nv_fp8_e4m3, 16, 64>>(const AttentionParams<half>&);
#endif

}  // namespace turbomind
