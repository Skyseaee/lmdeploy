// Copyright (c) OpenMMLab. All rights reserved.

#include "../decoding_config.h"
#include "../decoding_template.h"

namespace turbomind {

using namespace attention;

#ifdef ENABLE_FP8
template bool invokeDecoding<Decoding<arch::Sm80, half, fp8_dynamic, 8, 128>>(const AttentionParams<half>&);

template bool invokeDecoding<Decoding<arch::Sm80, half, fp8_dynamic, 16, 128>>(const AttentionParams<half>&);
#endif

}  // namespace turbomind
