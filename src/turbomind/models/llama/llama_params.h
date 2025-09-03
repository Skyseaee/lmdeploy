// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cstddef>
#include <map>
#include <regex>
#include <string>

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/quant_mode.h"
#include "src/turbomind/models/llama/llama_rope.h"

namespace turbomind {

// clang-format off
/* NOTE(Alan): MLA Param Parser
 *                  Input         [B, S, H]                                    = HIDDEN FEATURE    [B, S, H]
 *
 *    Q  Operator: Latent_Q      [B, S, Q_LORA_RANK]                          = Input * W_DownQ    [H, Q_LORA_RANK]
 *    Q  Operator: Latent_Up_Q   [B, S, NUM_HEAD * HEAD_DIM]                  = Latent_Q * W_UpQ   [Q_LORA_RANK, NUM_HEAD * HEAD_DIM]
 *    Q  Operator: Latent_Rope_Q [B, S, NUM_HEAD * QK_ROPE_DIM]               = Latent_Q * W_RopeQ [Q_LORA_RANK, NUM_HEAD * QK_ROPE_DIM]
 *    Q  Operator: RoPE_Q        [B, S, NUM_HEAD * QK_ROPE_DIM]               = RoPE(Latent_Rope_Q)
 *    Q  Operator: Concated_Q    [B, S, NUM_HEAD * (HEAD_DIM + QK_ROPE_DIM)]  = Concat(Latent_Up_Q, RoPE_Q)
 * 
 *    K  Operator: Latent_RoPE_K [B, S, QK_ROPE_DIM]                          = Input * W_RopeK  [H, QK_ROPE_DIM]  需要缓存   
 *    KV Operator: Latent_KV     [B, S, KV_LORA_RANK]                         = Input * W_DownKV [H, KV_LORA_RANK] 需要缓存
 *    K  Operator: Latent_Up_K   [B, S, NUM_HEAD * HEAD_DIM]                  = Latent_KV * W_UpK[KV_LORA_RANK, NUM_HEAD * HEAD_DIM]
 *    K  Operator: RoPE_K        [B, S, QK_ROPE_DIM]                          = RoPE(Latent_RoPE_K)
 *    K  Operator: Concated_K    [B, S, NUM_HEAD * (HEAD_DIM + QK_ROPE_DIM)]  = Concat(Latent_Up_K, RoPE_K)
 * 
 *    V  Operator: Latent_Up_V   [B, S, NUM_HEAD * V_HEAD_DIM]                = Latent_KV * W_UpV[KV_LORA_RANK, NUM_HEAD * V_HEAD_DIM]
 * 
 *    O  Operator: Prob          [B, NUM_HEAD, S, S]                          = Reshape(Concated_Q)[B, NUM_HEAD, S, (HEAD_DIM + QK_ROPE_DIM)] * Concated_K^T[B, NUM_HEAD, (HEAD_DIM + QK_ROPE_DIM), S]
 *    O  Operator: Score         [B, NUM_HEAD, S, S]                          = Softmax(Porb)
 *    O  Operator: Result        [B, NUM_HEAD, S, V_HEAD_DIM]                 = Score * (Latent_Up_V)[B, NUM_HEAD, S, V_HEAD_DIM] 
 *    O  Operator: Output        [B, S, NUM_HEAD * V_HEAD_DIM]                = Reshape(Result)
 */
// clang-format on
struct MLAParam {
    int q_lora_rank;
    int kv_lora_rank;
    int qk_rope_dim;
    int v_head_dim;
};

struct ModelParam {
    size_t   head_num;
    size_t   head_dim;
    size_t   kv_head_num;
    size_t   hidden_units;
    size_t   layer_num;
    size_t   vocab_size;
    size_t   embedding_size;
    size_t   tokenizer_size;
    float    norm_eps;
    int      quant_policy;
    bool     attn_bias;
    DataType weight_type;
    DataType data_type;
    int      group_size;
    MLAParam mla;
    bool     qk_norm;
    int      tune_layer_num;

    std::vector<int> inter_size;

    QuantMode quant_mode = QuantMode::fromDescription();
};

struct MoeParam {
    enum Method
    {
        kNaive,
        kFused
    } method;

    int   experts_per_token;
    int   inter_size;
    bool  norm_topk_prob;
    bool  shared_gate;
    float routed_scale;

    int         topk_group;
    std::string topk_method;
    int         n_group;

    std::vector<int> expert_num;

    bool enable_expert_pruning;
    int  keep_expert_num;
};

struct AttentionParam {
    float softmax_scale;
    int   cache_block_seq_len;
    // logn attention
    bool use_logn_attn;
    int  max_position_embeddings;
    // rotary embedding
    RopeParam rope;
};

struct EngineParam {
    // batch params
    int max_batch_size;
    int session_len;
    int step_length;

    // cache params
    float cache_max_block_count;
    int   cache_chunk_size;
    bool  enable_prefix_caching;

    // chunking params
    int max_forward_token_num;
    int max_context_token_num;
    int num_tokens_per_iter;
    int max_prefill_iters;

    // parallel params
    int outer_dp_size;
    int outer_dp_rank;
    int attn_dp_size;
    int attn_dp_rank;
    int attn_tp_size;
    int attn_tp_rank;
    int mlp_tp_size;
    int mlp_tp_rank;
    int moe_tp_size;
    int moe_tp_rank;
    int moe_ep_size;
    int moe_ep_rank;

    std::vector<int> devices;
};

enum class LoraPolicy : int
{
    kNull,
    kPlora,
};

struct LoraParam {
    int        r;
    float      scale;
    LoraPolicy policy;
    int        max_wo_r;

    std::map<std::string, std::pair<std::regex, int>>   rank_pattern;
    std::map<std::string, std::pair<std::regex, float>> scale_pattern;
};

}  // namespace turbomind
