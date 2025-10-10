#pragma once

#include <chrono>
#include <cstdint>
#include <ostream>

namespace turbomind {

struct ScheduleMetrics {
    // sequences
    int total_seqs;   // the number of received sequences
    int active_seqs;  // the number of active sequences
    int waiting_seqs; // the number of waiting sequences

    // kv block usage
    int total_blocks;  // the number of total kv blocks
    int active_blocks; // the number of active kv blocks
    int cached_blocks; // the number of cached kv blocks
    int free_blocks;   // the number of free kv blocks

    int decode_count;  // the number of decode requests
    int prefill_count; // the number of prefill requests

    double hit_rate;    // the hit rate of kv blocks for prefill-caching
    int cache_query_hit; // the number of cache hits
    int cache_query_total; // the number of cache queries
};

struct RequestMetrics {
    int64_t enque_time{};       // when a request is enqued
    int64_t scheduled_time{};   // when a request is scheduled for inference
    
    static int64_t timestamp()
    {
        // Get current timestamp in microseconds since Unix epoch
        // system_clock uses wall-clock time (matches Python's time.time())
        return std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now().time_since_epoch())
            .count();
    }
};

inline std::ostream& operator<<(std::ostream& os, const ScheduleMetrics& m)
{
    os << "ScheduleMetrics { "
       << ", total_seqs=" << m.total_seqs << ", "
       << ", active_seqs=" << m.active_seqs << ", "
       << ", waiting_seqs=" << m.waiting_seqs << ", "
       << ", total_blocks=" << m.total_blocks << ", "
       << ", cached_blocks=" << m.cached_blocks << ", "
       << ", active_blocks=" << m.active_blocks << ", "
       << ", hit_rate=" << m.hit_rate << ", "
       << ", free_blocks=" << m.free_blocks << ", "
       << ", decode_count=" << m.decode_count << ", "
       << ", prefill_count=" << m.prefill_count << ", "
       << ", cache_query_hit=" << m.cache_query_hit << ", "
       << ", cache_query_total=" << m.cache_query_total << " }";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const RequestMetrics& m)
{
    os << "RequestMetrics { "
       << ", enque_time=" << m.enque_time << ", "
       << ", scheduled_time=" << m.scheduled_time << " }";
    return os;
}

} // namespace turbomind