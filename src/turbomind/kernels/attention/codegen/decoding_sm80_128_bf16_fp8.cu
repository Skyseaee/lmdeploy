// Copyright (c) OpenMMLab. All rights reserved.

#include "../decoding_config.h"
#include "../decoding_template.h"

namespace turbomind {

using namespace attention;

#ifdef ENABLE_FP8
template bool invokeDecoding<Decoding<arch::Sm80, nv_bfloat16, __nv_fp8_e4m3, 8, 128>>(const AttentionParams<nv_bfloat16>&);

template bool invokeDecoding<Decoding<arch::Sm80, nv_bfloat16, __nv_fp8_e4m3, 16, 128>>(const AttentionParams<nv_bfloat16>&);
#endif

}  // namespace turbomind
