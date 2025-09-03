// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <istream>
#include <ostream>

#include "src/turbomind/core/core.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"

namespace turbomind {

class LlamaLinear {
public:
    enum Type
    {
        kGemm,
        kFusedSiluFfn,
        kFusedAdd
    };

    explicit LlamaLinear(cudaStream_t stream, const ModelParam& model, const EngineParam& engine, const MoeParam& moe);

    Tensor forward(const Tensor&           input,  //
                   const LlamaDenseWeight& weight,
                   Type                    type       = kGemm,
                   std::optional<Tensor>   output     = {},
                   void*                   param      = nullptr,
                   cudaStream_t            cur_stream = nullptr);

    void forward_moe(Tensor&                 output,
                     const Tensor&           input,
                     const int*              indexes,
                     const int*              offsets,
                     const LlamaDenseWeight& weight,
                     Type                    type,
                     gemm::Context*          context);

    void forward_cutlass_moe(Tensor&               output,
                             Tensor&               input,
                             Tensor&               logits,
                             //Tensor&               inter_buf_fp8,
                             Tensor&               cutlass_inout_buf,
                             const LlamaFfnWeight& weights,
                             int                   tokens,
                             int                   expert_num,
                             bool                  use_shared_stream,
                             cudaEvent_t           shared_expert_event,
                             cudaStream_t          shared_expert_stream);

    void set_measure(bool measure);

    [[maybe_unused]] int Export(std::ostream& os);

    [[maybe_unused]] int Import(std::istream& is);

    std::vector<int> GetTuningSeq() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace turbomind
