/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Modified from https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/layers/FfnLayer.cc

#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"

#ifdef FUSED_GATED_GEMM
#include "src/turbomind/kernels/fused_gated_gemm/fused_gated_gemm.h"
#include "src/turbomind/kernels/gemm_profiler/gemmPluginProfiler.h"
#include "src/turbomind/kernels/gemm_profiler/gemmSwigluProfiler.h"

namespace tlp = tensorrt_llm::plugins;
#endif

namespace turbomind {

class LlamaFfnLayer {
public:
    LlamaFfnLayer(const ModelParam& model, const Context& ctx): hidden_units_(model.hidden_units), linear_(*ctx.linear)
    {
        // TODO(Alan): currently not support TP/PP
        // Note(meng): We only config CUTLASS Gemm in fp8 mode
        // if(weight_type == turbomind::WeightType::kFP8)
        //     configGemm();
    }

    struct ForwardParam {
        Tensor                input;
        Tensor                output;
        const LlamaFfnWeight* weights;
        int                   layer_id;
        cudaStream_t          cur_stream = nullptr;
    };

    void forward(ForwardParam param);

private:
    void activation(Tensor& gating, Tensor& inter, cudaStream_t stream);

private:
    const size_t hidden_units_;
    LlamaLinear& linear_;
};

}  // namespace turbomind
