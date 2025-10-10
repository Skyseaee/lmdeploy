// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/models/llama/BlockManager.h"
#include <memory>
#include <unordered_map>
#include <vector>

namespace turbomind {

struct Sequence;

struct TrieNode {
    std::unordered_map<size_t, std::shared_ptr<TrieNode>> children;
    size_t                                                hash_key;
    std::vector<int>                                      tokens;
    int                                                   block_id;
    uint64_t                                              block_unique_id;
    int                                                   num_matched;
};

class BlockTrie {
public:
    explicit BlockTrie(size_t block_len_, std::shared_ptr<BlockManager> block_manager, bool enable_prefix_caching);

    bool enabled()
    {
        return enable_prefix_caching_;
    }

    // get cached blocks for sequence
    void match(Sequence& seq);

    // cache computed blocks for sequence
    void cache(const Sequence& seq);

    // remove invalid nodes, return valid count
    int verify();

    void query(int computed_blocks, int blocks_needed)
    {
        prefix_cache_stats_->requests += 1;
        prefix_cache_stats_->queries += blocks_needed;
        prefix_cache_stats_->hits += computed_blocks;
    }

    double hit_rate() noexcept
    {
        // prefix_caching_metrics_->observe(*prefix_cache_stats_);
        // prefix_cache_stats_ = std::make_shared<PrefixCacheStats>(0, 0, 0, false);
        return prefix_caching_metrics_->hit_rate();
    }

    void reset_prefix_cache()
    {
        // prefix_cache_stats_->reset = true;
        prefix_caching_metrics_->observe(*prefix_cache_stats_);
        prefix_cache_stats_ = std::make_shared<PrefixCacheStats>(0, 0, 0, false);
    }

    int cache_query_hit() const noexcept
    {
        return prefix_cache_stats_->hits;
    }

    int cache_query_total() const noexcept
    {
        return prefix_cache_stats_->queries;
    }

private:
    int verify_traverse(std::shared_ptr<TrieNode>& node);

private:
    bool   enable_prefix_caching_;
    size_t block_seq_len_;

    std::shared_ptr<PrefixCachingMetrics> prefix_caching_metrics_;
    std::shared_ptr<PrefixCacheStats>     prefix_cache_stats_;
    std::shared_ptr<BlockManager> block_manager_;

    std::shared_ptr<TrieNode> root_;
};

}  // namespace turbomind
