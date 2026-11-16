// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <chrono>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <string>

#include "algorithm/sindi/sindi_cpu_timer.h"
#include "algorithm/sindi_v2/sindi_v2_parameter.h"
#include "datacell/sindi_search_term_datacell.h"
#include "query_context.h"

namespace vsag {

class SearchExecutionContext;

struct SindiV2SearchProfile {
    using Clock = std::chrono::steady_clock;
    bool enabled{false};
    bool diagnose{false};
    bool profile_cpu{false};
    SindiCpuTime caller_cpu;
    uint64_t compute_cpu_ns{0};
    bool compute_cpu_complete{true};
    bool parallel{false};
    uint32_t requested{1};
    uint32_t granted{1};
    const char* fallback{"disabled"};
    const char* stop{"complete"};
    Clock::time_point start{Clock::now()};
    Clock::time_point deadline{Clock::time_point::max()};
    uint64_t prepare_ns{0};
    uint64_t batch_ns{0};
    uint64_t caller_compute_ns{0};
    uint64_t wait_ns{0};
    uint64_t compute_sum_ns{0};
    uint64_t compute_max_ns{0};
    std::array<uint64_t, 6> compute_histogram{};
    uint32_t slowest_window{0};
    uint64_t slowest_window_postings{0};
    uint64_t slowest_batch_ns{0};
    uint64_t slowest_batch_wait_ns{0};
    uint32_t slowest_batch_first{0};
    uint32_t slowest_batch_last{0};
    uint32_t slowest_batch_width{0};
    uint64_t consume_ns{0};
    uint64_t rerank_ns{0};
    uint64_t output_ns{0};
    uint64_t batches{0};
    uint64_t computed_windows{0};
    uint64_t consumed_windows{0};
    uint64_t candidate_count{0};
    uint64_t query_terms{0};
    uint64_t scan_postings{0};
    bool scan_postings_available{false};
    uint64_t window_digest{14695981039346656037ULL};
    uint64_t candidate_digest{14695981039346656037ULL};
    uint64_t result_digest{14695981039346656037ULL};
    SparseQueryProfile rerank;

    void
    CheckDeadline() const;
    void
    RecordCompute(uint32_t window, uint64_t elapsed_ns, uint64_t postings);
    std::string
    Dump(const SearchStatistics& statistics) const;
    void
    DigestWindow(uint32_t window, const Vector<float>& dists);
    void
    DigestCandidates(const MaxHeap& heap);
    void
    DigestResults(const DatasetPtr& result);
};

struct SindiV2WindowSlot {
    SindiV2WindowSlot(const SparseTermComputerPtr& query, uint32_t window_size);
    std::shared_ptr<Allocator> allocator;
    SparseTermComputerPtr computer;
    Vector<float> dists;
    SindiQueryContext context;
    uint32_t window{0};
    bool prune{true};
    std::exception_ptr error;
    bool started{false};
    uint64_t compute_ns{0};
    SindiCpuTime cpu;
};

SindiV2SearchProfile
MakeSindiV2Profile(const SINDIV2SearchParameter& parameters,
                   SindiV2SearchProfile::Clock::time_point start);

// Errors cannot carry a Dataset. Emit the same bounded, payload-free diagnostics
// when profiling is enabled, while preserving the original exception.
class SindiV2FailureDiagnostics {
public:
    SindiV2FailureDiagnostics(SindiV2SearchProfile& profile, SearchStatistics& statistics)
        : profile_(profile), statistics_(statistics), exceptions_(std::uncaught_exceptions()) {
    }
    ~SindiV2FailureDiagnostics();

private:
    SindiV2SearchProfile& profile_;
    SearchStatistics& statistics_;
    int exceptions_;
};

// next() chooses windows on the calling thread. consume() alone touches filters
// and the global heap, in that same order. False ends the successful prefix.
void
SearchSindiV2Batches(SearchExecutionContext& resources,
                     const SindiSearchTermDataCellPtr& datacell,
                     const SparseTermComputerPtr& computer,
                     uint32_t window_size,
                     bool use_term_lists,
                     const std::function<bool(uint32_t&, bool&)>& next,
                     const std::function<bool(SindiV2WindowSlot&)>& consume,
                     SindiV2SearchProfile& profile,
                     SearchStatistics* statistics);

}  // namespace vsag
