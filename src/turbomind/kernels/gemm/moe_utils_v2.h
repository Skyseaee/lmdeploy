// Copyright (c) OpenMMLab. All rights reserved.

#include <cstdint>
#include <cuda_runtime.h>
#include <random>
#include <vector>

#include "src/turbomind/core/core.h"

namespace turbomind {

constexpr int kMoeGateMaxTiles = 16;
constexpr int kMoeGateVecSize  = 4;

void invokeMoeGate_V2(int*         f2n,
                      int*         en2f,
                      int*         offsets,
                      float*       scales,
                      void*        masks,
                      int*         accum,
                      const float* logits,
                      int          tokens,
                      int          tokens_padded,
                      int          experts,
                      int          exp_per_tok,
                      bool         softmax,
                      bool         norm_topk,
                      float        routed_scale,
                      int2         expert_range,
                      cudaStream_t st);

void invokeMoeDispatch(Ref<Tensor>   out_,  //
                       const Tensor& src,
                       const int*    f2n,
                       int           expert_per_token,
                       cudaStream_t  st);

void invokeMaskExpertsByVoteFusedV2(
    float*       logits,
    int*         votes,
    int*         hists,
    int          tokens,
    int          expert_num,
    int          top_k,
    int          keep_expert_num,
    cudaStream_t stream
);

template<class T>
void invokeMoeGather(
    T* dst, const T* src, const int* f2n, int tokens, int experts_per_token, int dims, cudaStream_t st);

void invokeMoeDispatchScales(Ref<Tensor>   out_,  //
                             const Tensor& src,
                             const int*    f2n,
                             int           expert_per_token,
                             cudaStream_t  st);

void invokeMoeCombine(Ref<Tensor>   out_,
                      const Tensor& src,
                      const float*  scales,
                      const int*    en2f,
                      const float*  dst_scales,
                      int           experts_per_token,
                      float         dst_scale,
                      cudaStream_t  st);

void invokeMoeSoftmaxMaskTopKGroups(
    float* logits, int token_num, int expert_num, int group_size, int top_k, cudaStream_t st);

template<class T>
void invokeMoeReduce(T*           dst,
                     const T*     src,
                     const float* scales,
                     const int*   en2f,
                     const float* dst_scales,
                     int          tokens,
                     int          experts_per_token,
                     int          dims,
                     float        dst_scale,
                     cudaStream_t st);

template<class T>
void invokeFusedMoeReduce(
    T* dst, const T* src, const float* dst_scales, int tokens, int dims, float dst_scale, cudaStream_t st);

void invokeMaskMoeTopKGroups(float* logits, int token_num, int expert_num, int group_size, int top_k, cudaStream_t st);

// Sample `e` from `E` experts uniformly for every token
std::vector<int> SampleUniform(int token_num, int expert_num, int exp_per_tok, std::mt19937& g);

std::vector<int> SampleBalanced(int token_num, int expert_num, int exp_per_tok, std::mt19937& g);

void invokeMoveOffsets(int* offsets, int expert, int2 expert_range, cudaStream_t st);

}  // namespace turbomind
