// Copyright (c) OpenMMLab. All rights reserved.

#include "../decoding_config.h"
#include "../decoding_template.h"

namespace turbomind {

using namespace attention;

#ifdef ENABLE_FP8
template bool invokeDecoding<Decoding<arch::Sm80, nv_bfloat16, fp8_dynamic, 8, 64>>(const AttentionParams<nv_bfloat16>&);

template bool invokeDecoding<Decoding<arch::Sm80, nv_bfloat16, fp8_dynamic, 16, 64>>(const AttentionParams<nv_bfloat16>&);
#endif

}  // namespace turbomind
