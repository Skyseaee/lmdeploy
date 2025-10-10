#pragma once

#include <cstdint>
#include <deque>
#include <iostream>
#include <ostream>
#include <tuple>

namespace turbomind {

/**
 * A utility class to maintain cache metric.
 * To avoid overflow, we maintain the hit rate in block granularity, so that
 * we can maintain a single hit rate for n_completed_block x block_size,
 * and calculate the real time hit rate by the following:
 * BS = The number of queries per block.
 * nB = The number of completed blocks.
 * HR = hit rate of (nB x BS) queries.
 * Q = current number of queries (< BS).
 * H = current number of hits (< BS).
 * hit rate = ((HR x nB) + (H / Q) x (Q / BS)) / (nB + Q / BS)
 */
class CacheMetricsData {
public:
    CacheMetricsData(int block_size = 1000)
        : num_completed_blocks(0)
        , completed_block_cache_hit_rate(0.0)
        , num_incompleted_block_queries(0)
        , num_incompleted_block_hit(0)
        , block_size(block_size)
    {
    }

    /**
     * Record a query result
     * @param hit Whether the query was a cache hit
     */
    void query(bool hit);

    /**
     * Get the current hit rate
     * @return The calculated hit rate as a double
     */
    double get_hit_rate() const;

    friend std::ostream& operator<<(std::ostream& os, const CacheMetricsData& m);

private:
    int num_completed_blocks;                    // Number of completed blocks
    double completed_block_cache_hit_rate;       // Hit rate of completed blocks
    int num_incompleted_block_queries;           // Number of queries in current incomplete block
    int num_incompleted_block_hit;               // Number of hits in current incomplete block
    int block_size;                              // Size of each block
};

inline void CacheMetricsData::query(bool hit)
{
    num_incompleted_block_queries += 1;
    num_incompleted_block_hit += hit ? 1 : 0;

    // When a block is completed, update the cache hit rate
    // and reset the incomplete numbers.
    if (num_incompleted_block_queries == block_size) {
        double hit_rate = static_cast<double>(num_incompleted_block_hit) /
                         static_cast<double>(num_incompleted_block_queries);
        completed_block_cache_hit_rate = 
            (completed_block_cache_hit_rate * num_completed_blocks + hit_rate) / 
            (num_completed_blocks + 1);
        num_incompleted_block_queries = 0;
        num_incompleted_block_hit = 0;
        num_completed_blocks += 1;
    }
}

inline double CacheMetricsData::get_hit_rate() const
{
    double incomplete_ratio = static_cast<double>(num_incompleted_block_queries) / 
                             static_cast<double>(block_size);
    double total_blocks = num_completed_blocks + incomplete_ratio;
    if (total_blocks == 0.0) {
        return 0.0;
    }

    double completed_block_hit = 0.0;
    double incompleted_block_hit = 0.0;
    
    if (num_completed_blocks > 0) {
        completed_block_hit = completed_block_cache_hit_rate * num_completed_blocks;
    }
    
    if (num_incompleted_block_queries > 0) {
        double incompleted_hit_rate = static_cast<double>(num_incompleted_block_hit) /
                                     static_cast<double>(num_incompleted_block_queries);
        incompleted_block_hit = incompleted_hit_rate * incomplete_ratio;
    }
    
    return (completed_block_hit + incompleted_block_hit) / total_blocks;
}

inline std::ostream& operator<<(std::ostream& os, const CacheMetricsData& m)
{
    os << "CacheMetricsData { "
       << ", num_completed_blocks=" << m.num_completed_blocks << ", "
       << ", completed_block_cache_hit_rate=" << m.completed_block_cache_hit_rate << ", "
       << ", num_incompleted_block_queries=" << m.num_incompleted_block_queries << ", "
       << ", num_incompleted_block_hit=" << m.num_incompleted_block_hit << ", "
       << ", block_size=" << m.block_size << " }";
    return os;
}

struct PrefixCacheStats {
    int requests;
    int queries;
    int hits;
    bool reset;

    PrefixCacheStats(int req = 0, int qry = 0, int hit = 0, bool r = false)
        : requests(req), queries(qry), hits(hit), reset(r) {}
};

class PrefixCachingMetrics {
public:
    /**
     * @brief Construct a new PrefixCachingMetrics object.
     * @param max_recent_requests The number of the max recent requests to aggregate. Defaults to 1000.
     */
    explicit PrefixCachingMetrics(int max_recent_requests = 1000)
        : max_recent_requests_(max_recent_requests)
        , aggregated_requests_(0)
        , aggregated_query_total_(0)
        , aggregated_query_hit_(0)
    {
    }

    /**
     * @brief Observe the prefix caching for a set of requests.
     * This function is called with information gathered when new requests
     * are being scheduled and are looking for computed blocks.
     *
     * When there are more than `max_recent_requests`, the oldest set of
     * requests are removed from the metrics.
     *
     * @param stats The prefix cache stats.
     */
    void observe(const PrefixCacheStats& stats);
    
    /**
     * @brief Reset the metrics.
     */
    void reset();

    /**
     * @brief Get the hit rate for the past N aggregated requests.
     * @return The hit rate as a float. Returns 0.0 if no queries have been made.
     */
    float hit_rate() const;

    /**
     * @brief Get the current number of aggregated requests.
     * @return int
     */
    int aggregated_requests_count() const { return aggregated_requests_; }

    /**
     * @brief Get the maximum number of recent requests tracked.
     * @return int
     */
    int max_recent_requests() const { return max_recent_requests_; }

    /**
     * @brief Get the current number of aggregated query hits.
     * @return int
     */
    int aggregated_query_hit() const { return aggregated_query_hit_; }

    /**
     * @brief Get the current number of aggregated query total.
     * @return int
     */
    int aggregated_query_total() const { return aggregated_query_total_; }

private:
    int max_recent_requests_;           // Max number of recent requests to track
    int aggregated_requests_;           // Total number of requests in current window
    int aggregated_query_total_;        // Total number of queries in current window
    int aggregated_query_hit_;          // Total number of query hits in current window

    // Queue of (requests, queries, hits) for the most recent stats
    std::deque<std::tuple<int, int, int>> query_queue_;
};

inline void PrefixCachingMetrics::observe(const PrefixCacheStats& stats) {
    // reset_prefix_cache was invoked before the current update.
    // Reset the metrics before aggregating the current stats.
    if (stats.reset) {
        reset();
    }

    // Update the metrics
    query_queue_.emplace_back(stats.requests, stats.queries, stats.hits);
    aggregated_requests_ += stats.requests;
    aggregated_query_total_ += stats.queries;
    aggregated_query_hit_ += stats.hits;

    // Remove the oldest stats if the number of requests exceeds the max recent requests.
    if (aggregated_requests_ > max_recent_requests_) {
        const auto& old_stats = query_queue_.front();
        aggregated_requests_ -= std::get<0>(old_stats);
        aggregated_query_total_ -= std::get<1>(old_stats);
        aggregated_query_hit_ -= std::get<2>(old_stats);
        query_queue_.pop_front();
    }
}

inline void PrefixCachingMetrics::reset() {
    aggregated_requests_ = 0;
    aggregated_query_total_ = 0;
    aggregated_query_hit_ = 0;
    query_queue_.clear();
}

inline float PrefixCachingMetrics::hit_rate() const {
    if (aggregated_requests_ == 0) {
        return 0.0f;
    }

    return static_cast<float>(aggregated_query_hit_) / aggregated_query_total_;
}
    
} // namespace turbomind