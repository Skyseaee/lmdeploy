// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/BlockTrie.h"
#include "src/turbomind/models/llama/SequenceManager.h"

namespace turbomind {

size_t hash(const std::vector<int>& vec)
{
    size_t seed = vec.size();
    for (const auto& i : vec) {
        seed ^= std::hash<int>{}(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

BlockTrie::BlockTrie(size_t block_len, std::shared_ptr<BlockManager> block_manager):
    block_seq_len_(block_len), block_manager_(block_manager)
{
    root_ = std::make_shared<TrieNode>();
    if (enable_prefix_caching_) {
        prefix_caching_metrics_ = std::make_shared<PrefixCachingMetrics>(1000);
        prefix_cache_stats_ = std::make_shared<PrefixCacheStats>(0, 0, 0, false);
    }
}

std::tuple<BlockIds, UniqueIds> BlockTrie::MatchPrefix(const Sequence& seq)
{
    BlockIds  matched_blocks;
    UniqueIds matched_unique_ids;

    std::shared_ptr<TrieNode> curr_node   = root_;
    int                       num_matched = 0;

    // Warning: Do not use "<=" operator even when seq.prompt length is evenly
    // divisible by block_seq_len_. This may produce an input_length of zero for
    // the sequence, violating the precondition checked in LlamaBatch::Forward.
    while (num_matched + block_seq_len_ < seq.prompt.size()) {
        std::vector<int> curr_tokens(seq.prompt.begin() + num_matched,
                                     seq.prompt.begin() + num_matched + block_seq_len_);
        size_t           hash_key = hash(curr_tokens);

        auto it = curr_node->children.find(hash_key);

        if (it == curr_node->children.end()) {
            break;
        }

        if (curr_tokens != it->second->tokens) {
            TM_LOG_WARNING("hash key cache hit, but tokens are not the same");
            break;
        }

        matched_blocks.emplace_back(it->second->block_id);
        matched_unique_ids.emplace_back(it->second->block_unique_id);
        curr_node = it->second;
        num_matched += block_seq_len_;
    }

    if (matched_blocks.size() > 0) {
        // add use count
        block_manager_->Lock(matched_blocks);
        block_manager_->Touch(matched_blocks);
        // only consider no history blocks
        seq.blocks.insert(seq.blocks.end(), matched_blocks.begin(), matched_blocks.end());
        seq.block_unique_ids.insert(seq.block_unique_ids.end(), matched_unique_ids.begin(), matched_unique_ids.end());
    }

    // query cache metrics data
    int total_blocks_needed = seq.prompt.size() / block_seq_len_;

    int num_hits = static_cast<int>(matched_blocks.size());
    FT_CHECK(num_hits <= total_blocks_needed);

    if (seq.block_trie_matched) {
        total_blocks_needed = 0; // no need to match more
    } else {
        seq.block_trie_matched = true;
    }

    query(num_hits, total_blocks_needed);
}

void BlockTrie::cache(const Sequence& seq)
{
    FT_CHECK(seq.status != Sequence::kCached);
    FT_CHECK(tokens.size() <= seq.blocks.size() * block_seq_len_);

    std::shared_ptr<TrieNode> curr_node = root_;
    int                             idx = 0;

    BlockIds cache_block_ids;
    UniqueIds cache_block_unique_ids;

    // We don't cache the last block of the sequence, since it might not be full
    // TODO(lvhan): determine wether the last block is full or not. It is not trivial
    // considering chunk prefill
    for (int idx=0; idx < (int)seq.blocks.size() - 1; ++idx) {
        auto start = tokens.begin() + idx * block_seq_len_;
        auto end = start + block_seq_len_;
        
        std::vector<int> curr_tokens(start, end);
        size_t hash_key = hash(curr_tokens);

        int block_id = seq.blocks[idx];
        uint64_t block_unique_id = seq.block_unique_ids[idx];

        auto it = curr_node->children.find(hash_key);
        if (it != curr_node->children.end()) {
            if (curr_tokens != it->second->tokens) {
                TM_LOG_WARNING("hash key cache hit, but tokens are not the same");
                break;
            }
            curr_node = it->second;
            curr_node->block_id = block_id;
            curr_node->block_unique_id = block_unique_id;
        }
        else {
            std::shared_ptr<TrieNode> node = std::make_shared<TrieNode>();
            node->hash_key = hash_key;
            node->tokens = curr_tokens;
            node->block_id = block_id;
            node->block_unique_id = block_unique_id;
            curr_node->children[hash_key] = node;
            curr_node = node;
        }

        cache_block_ids.emplace_back(block_id);
        cache_block_unique_ids.emplace_back(block_unique_id);
    }

    return std::make_tuple(cache_block_ids, cache_block_unique_ids);
}

void BlockTrie::Verify()
{
    DFS(root_);
}

void BlockTrie::DFS(std::shared_ptr<TrieNode>& node)
{
    for (auto it = node->children.begin(); it != node->children.end();) {
        if (block_manager_->unique_id(it->second->block_id) != it->second->block_unique_id) {
            // child invalid
            it = node->children.erase(it);
        }
        else {
            DFS(it->second);
            ++it;
        }
    }
}

// void BlockTrie::_DFS(std::shared_ptr<TrieNode>& node)
// {
//     for (auto it = node->children.begin(); it != node->children.end();) {
//         bool valid = true;
//         for (int i = 0; i < it->second->block_ids.size(); ++i) {
//             auto block_id = it->second->block_ids[i];
//             if (block_manager_->unique_id(block_id) != it->second->block_unique_ids[i]) {
//                 it = node->children.erase(it);
//                 valid = false;
//                 break;
//             }
//         }

//         if (valid) {
//             _DFS(it->second);
//             ++it;
//         }
//     }
// }

// std::tuple<BlockIds, UniqueIds> BlockTrie::MatchPrefix(const Sequence& seq)
// {
//     BlockIds  matched_blocks;
//     UniqueIds matched_unique_ids;

//     std::shared_ptr<TrieNode> curr_node   = root_;
//     int                       num_matched = 0;

//     // Warning: Do not use "<=" operator even when seq.prompt length is evenly
//     // divisible by block_seq_len_. This may produce an input_length of zero for
//     // the sequence, violating the precondition checked in LlamaBatch::Forward.
//     while (num_matched + block_seq_len_ < seq.prompt.size()) {
//         std::vector<int> curr_tokens(seq.prompt.begin() + num_matched,
//                                      seq.prompt.begin() + num_matched + block_seq_len_);
//         size_t           hash_key = hash(curr_tokens);

//         auto it = curr_node->children.find(hash_key);

//         if (it == curr_node->children.end()) {
//             break;
//         }

//         if (curr_tokens != it->second->tokens) {
//             TM_LOG_WARNING("hash key cache hit, but tokens are not the same");
//             break;
//         }

//         matched_blocks.emplace_back(it->second->block_id);
//         matched_unique_ids.emplace_back(it->second->block_unique_id);
//         curr_node = it->second;
//         num_matched += block_seq_len_;
//     }

//     return std::make_tuple(matched_blocks, matched_unique_ids);
// }

// std::tuple<BlockIds, UniqueIds> BlockTrie::Insert(const Sequence& seq, const std::vector<int>& tokens)
// {
//     FT_CHECK(seq.status != Sequence::kCached);
//     FT_CHECK(tokens.size() <= seq.blocks.size() * block_seq_len_);

//     std::shared_ptr<TrieNode> curr_node = root_;
//     int                             idx = 0;

//     BlockIds cache_block_ids;
//     UniqueIds cache_block_unique_ids;

//     // We don't cache the last block of the sequence, since it might not be full
//     // TODO(lvhan): determine wether the last block is full or not. It is not trivial
//     // considering chunk prefill
//     for (int idx=0; idx < (int)seq.blocks.size() - 1; ++idx) {
//         auto start = tokens.begin() + idx * block_seq_len_;
//         auto end = start + block_seq_len_;
        
//         std::vector<int> curr_tokens(start, end);
//         size_t hash_key = hash(curr_tokens);

//         int block_id = seq.blocks[idx];
//         uint64_t block_unique_id = seq.block_unique_ids[idx];

//         auto it = curr_node->children.find(hash_key);
//         if (it != curr_node->children.end()) {
//             if (curr_tokens != it->second->tokens) {
//                 TM_LOG_WARNING("hash key cache hit, but tokens are not the same");
//                 break;
//             }
//             curr_node = it->second;
//             curr_node->block_id = block_id;
//             curr_node->block_unique_id = block_unique_id;
//         }
//         else {
//             std::shared_ptr<TrieNode> node = std::make_shared<TrieNode>();
//             node->hash_key = hash_key;
//             node->tokens = curr_tokens;
//             node->block_id = block_id;
//             node->block_unique_id = block_unique_id;
//             curr_node->children[hash_key] = node;
//             curr_node = node;
//         }

//         cache_block_ids.emplace_back(block_id);
//         cache_block_unique_ids.emplace_back(block_unique_id);
//     }

//     return std::make_tuple(cache_block_ids, cache_block_unique_ids);
// }

void BlockTrie::Verify()
{
    // DFS(root_);
    _DFS(root_);
}

// void BlockTrie::DFS(std::shared_ptr<TrieNode>& node)
// {
//     for (auto it = node->children.begin(); it != node->children.end();) {
//         if (block_manager_->unique_id(it->second->block_id) != it->second->block_unique_id) {
//             // child invalid
//             it = node->children.erase(it);
//         }
//         else {
//             DFS(it->second);
//             ++it;
//         }
//     }
// }

void BlockTrie::_DFS(std::shared_ptr<TrieNode>& node)
{
    for (auto it = node->children.begin(); it != node->children.end();) {
        bool valid = true;
        for (int i = 0; i < it->second->block_ids.size(); ++i) {
            auto block_id = it->second->block_ids[i];
            if (block_manager_->unique_id(block_id) != it->second->block_unique_ids[i]) {
                // split_child(it->second, it->second->tokens, it->second->tokens.size());
                it = node->children.erase(it);
                valid = false;
                break;
            }
        }

        if (valid) {
            _DFS(it->second);
            ++it;
        }
    }
}

std::tuple<BlockIds, UniqueIds> BlockTrie::MatchPrefix(const Sequence& seq)
{
    if (seq.tokens.size() == 0) {
        return std::make_tuple(BlockIds(), UniqueIds());
    }

    int page_aligned_len = seq.tokens.size() / block_seq_len_ * block_seq_len_;
    std::vector<int> page_aligned_tokens(seq.tokens.begin(), seq.tokens.begin() + page_aligned_len);

    return _match_prefix_helper(root_, page_aligned_tokens);
}

std::tuple<BlockIds, UniqueIds> BlockTrie::Insert(const Sequence& seq, const std::vector<int>& tokens)
{
    std::vector<int> tokens_copy(tokens);
    return _insert_helper(root_, seq, tokens_copy);
}

std::tuple<BlockIds, UniqueIds> BlockTrie::_match_prefix_helper(std::shared_ptr<TrieNode>& node, std::vector<int>& tokens)
{
    node->last_access_time = get_timestamp();
    size_t keys = _get_keys(tokens);
    BlockIds  matched_blocks;
    UniqueIds matched_unique_ids;

    while (tokens.size() > 0 && node->children[keys]) {
        if (!tokens_equal(tokens, node->children[keys]->tokens)) {
            break;
        }
        auto child = node->children[keys];
        child->last_access_time = get_timestamp();
        int prefix_len = _key_match_page(child->tokens, tokens);
        int block_count = prefix_len / block_seq_len_;
        if (prefix_len < child->tokens.size()) {
            // split child
            auto new_node = _split_child(child, child->tokens, prefix_len);
            matched_blocks.insert(matched_blocks.end(), new_node->block_ids.begin(), new_node->block_ids.begin() + block_count);
            matched_unique_ids.insert(matched_unique_ids.end(), new_node->block_unique_ids.begin(), new_node->block_unique_ids.begin() + block_count);
            break;
        } else {
            node = child;
            tokens.erase(tokens.begin(), tokens.begin() + prefix_len);
            matched_blocks.insert(matched_blocks.end(), child->block_ids.begin(), child->block_ids.begin() + block_count);
            matched_unique_ids.insert(matched_unique_ids.end(), child->block_unique_ids.begin(), child->block_unique_ids.begin() + block_count);

            if (tokens.size() != 0) {
                keys = _get_keys(tokens);
            }
        }
    }

    return std::make_tuple(matched_blocks, matched_unique_ids);
}

bool BlockTrie::tokens_equal(const std::vector<int>& tokens0, const std::vector<int>& tokens1)
{
    for (int i = 0; i < block_seq_len_; ++i) {
        if (tokens0[i] != tokens1[i]) {
            return false;
        }
    }
    return true;
}

std::shared_ptr<TrieNode> BlockTrie::_split_child(std::shared_ptr<TrieNode>& child, std::vector<int>& keys, int split_len)
{
    FT_CHECK(split_len % block_seq_len_ == 0);
    int block_count = split_len / block_seq_len_;

    auto new_child = std::make_shared<TrieNode>();
    std::vector<int> child_tokens(keys.begin() + split_len, keys.end());
    size_t child_keys = _get_keys(child_tokens);

    new_child->block_ids = std::vector<int>(child->block_ids.begin(), child->block_ids.begin() + block_count);
    new_child->block_unique_ids = std::vector<uint64_t>(child->block_unique_ids.begin(), child->block_unique_ids.begin() + block_count);
    new_child->parent = child->parent;
    new_child->tokens = std::vector<int>(keys.begin(), keys.begin() + split_len);
    new_child->last_access_time = get_timestamp();

    auto parent = child->parent;
    auto key = _get_keys(new_child->tokens);
    parent->children[key] = new_child;
    new_child->hash_key = key;

    child->parent = new_child;
    child->tokens = child_tokens;
    child->block_ids.erase(child->block_ids.begin(), child->block_ids.begin() + block_count);
    child->block_unique_ids.erase(child->block_unique_ids.begin(), child->block_unique_ids.begin() + block_count);
    child->last_access_time = get_timestamp();
    new_child->children[child_keys] = child;
    return new_child;
}

int BlockTrie::_key_match_page(const std::vector<int>& keys0, const std::vector<int>& keys1)
{
    if (block_seq_len_ == 1) {
        int pos = 0;
        while (pos < keys0.size() && pos < keys1.size()) {
            if (keys0[pos] != keys1[pos]) {
                return pos;
            }
            pos++;
        }
        return pos;
    } else {
        int min_len = std::min(keys0.size(), keys1.size());
        for (int i = 0; i < min_len; i += block_seq_len_) {
            bool match = true;
            for (int j = 0; j < block_seq_len_; j++) {
                if (keys0[i + j] != keys1[i + j]) {
                    match = false;
                    break;
                }
            }
            if (!match) {
                return (i / block_seq_len_) * block_seq_len_;
            }
        }
        return (min_len / block_seq_len_) * block_seq_len_;
    }
}

size_t BlockTrie::_get_keys(const std::vector<int>& tokens)
{
    size_t seed = block_seq_len_;
    for (size_t i = 0; i < block_seq_len_; ++i) {
        seed ^= std::hash<int>{}(tokens[i]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

std::tuple<BlockIds, UniqueIds> BlockTrie::_insert_helper(std::shared_ptr<TrieNode>& node, const Sequence& seq, std::vector<int>& tokens)
{
    node->last_access_time = get_timestamp();
    if (seq.tokens.size() == 0) {
        return std::make_tuple(BlockIds(), UniqueIds());
    }

    FT_CHECK(tokens.size() <= seq.blocks.size() * block_seq_len_);

    if (seq.status == Sequence::kCached) {
        return std::make_tuple(seq.blocks, seq.block_unique_ids);
    }

    auto hash_key = _get_keys(tokens);
    BlockIds cache_block_ids;
    UniqueIds cache_block_unique_ids;

    int total_matched = 0;
    while (tokens.size() > 0 && node->children.find(hash_key) != node->children.end()) {
        if (!tokens_equal(tokens, node->children[hash_key]->tokens)) {
            break;
        }

        auto child = node->children[hash_key];
        child->last_access_time = get_timestamp();
        int prefix_len = _key_match_page(child->tokens, tokens);
        total_matched += prefix_len;
        tokens.erase(tokens.begin(), tokens.begin() + prefix_len);
        cache_block_ids.insert(cache_block_ids.end(), child->block_ids.begin(), child->block_ids.begin() + prefix_len / block_seq_len_);
        cache_block_unique_ids.insert(cache_block_unique_ids.end(), child->block_unique_ids.begin(), child->block_unique_ids.begin() + prefix_len / block_seq_len_);

        if (prefix_len < child->tokens.size()) {
            // split child
            auto new_node = _split_child(child, child->tokens, prefix_len);
            node = new_node;
        }

        if (tokens.size() != 0) {
            hash_key = _get_keys(tokens);
        }
    }

    if (tokens.size() != 0) {
        auto new_node = std::make_shared<TrieNode>();
        new_node->parent = node;
        new_node->tokens = tokens;
        new_node->hash_key = hash_key;
        new_node->block_ids = std::vector<int>(seq.blocks.begin() + total_matched / block_seq_len_, seq.blocks.begin() + (total_matched + tokens.size()) / block_seq_len_);
        new_node->block_unique_ids = std::vector<uint64_t>(seq.block_unique_ids.begin() + total_matched / block_seq_len_, seq.block_unique_ids.begin() + (total_matched + tokens.size()) / block_seq_len_);
        node->children[hash_key] = new_node;
        cache_block_ids.insert(cache_block_ids.end(), new_node->block_ids.begin(), new_node->block_ids.end());
        cache_block_unique_ids.insert(cache_block_unique_ids.end(), new_node->block_unique_ids.begin(), new_node->block_unique_ids.end());
    }

    return std::make_tuple(cache_block_ids, cache_block_unique_ids);
}

}  // namespace turbomind
