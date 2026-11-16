// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include "sindi_v2_parallel_search.h"

#include <algorithm>
#include <cmath>
#include <future>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "impl/logger/logger.h"
#include "impl/searcher/search_execution_context.h"
#include "vsag_exception.h"

namespace vsag {
namespace {
void
hash_word(uint64_t& hash, uint64_t word) {
    for (uint32_t i = 0; i < 8; ++i) {
        hash = (hash ^ (word & 255)) * 1099511628211ULL;
        word >>= 8;
    }
}
}  // namespace

SindiV2SearchProfile
make_sindi_v2_profile(const SINDIV2SearchParameter& parameters,
                      SindiV2SearchProfile::Clock::time_point start) {
    SindiV2SearchProfile profile;
    profile.start = start;
    profile.enabled = parameters.profile_search;
    profile.diagnose = parameters.diagnose_search;
    profile.profile_cpu = parameters.profile_cpu;
    profile.requested = parameters.window_parallelism;
    if (parameters.enable_time_record && profile.requested > 1) {
        const auto timeout = parameters.timeout_ms;
        CHECK_ARGUMENT(std::isfinite(timeout), "timeout_ms must be finite");
        const auto maximum = std::chrono::duration<double, std::milli>(
                                 SindiV2SearchProfile::Clock::time_point::max() - start)
                                 .count();
        if (timeout <= 0) {
            profile.deadline = start;
        } else if (timeout < maximum) {
            profile.deadline =
                start + std::chrono::duration_cast<SindiV2SearchProfile::Clock::duration>(
                            std::chrono::duration<double, std::milli>(timeout));
        }
    }
    return profile;
}

void
SindiV2SearchProfile::CheckDeadline() const {
    if (parallel && deadline != Clock::time_point::max() && Clock::now() >= deadline) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "SINDI_V2 parallel deadline exceeded");
    }
}

void
SindiV2SearchProfile::RecordCompute(uint32_t window, uint64_t elapsed_ns, uint64_t postings) {
    if (!enabled) {
        return;
    }
    constexpr std::array<uint64_t, 5> bounds{50000, 200000, 1000000, 5000000, 20000000};
    const auto bucket = std::upper_bound(bounds.begin(), bounds.end(), elapsed_ns) - bounds.begin();
    ++compute_histogram[bucket];
    compute_sum_ns += elapsed_ns;
    if (elapsed_ns >= compute_max_ns) {
        compute_max_ns = elapsed_ns;
        slowest_window = window;
        slowest_window_postings = postings;
    }
}

SindiV2FailureDiagnostics::~SindiV2FailureDiagnostics() {
    if (std::uncaught_exceptions() <= exceptions_ || !profile_.enabled) {
        return;
    }
    try {
        const bool timeout =
            profile_.parallel && SindiV2SearchProfile::Clock::now() >= profile_.deadline;
        profile_.stop = timeout ? "deadline" : "error";
        if (timeout) {
            statistics_.is_timeout.store(true, std::memory_order_relaxed);
        }
        logger::error("SINDI_V2 execution failure " + profile_.Dump(statistics_));
    } catch (...) {  // NOLINT(bugprone-empty-catch): preserve the original query exception.
        // Diagnostic allocation/logging must never replace the original failure.
    }
}

void
SindiV2SearchProfile::DigestWindow(uint32_t window, const Vector<float>& dists) {
    if (!diagnose) {
        return;
    }
    hash_word(window_digest, window);
    for (float distance : dists) {
        uint32_t bits;
        std::memcpy(&bits, &distance, sizeof(bits));
        hash_word(window_digest, bits);
    }
}

void
SindiV2SearchProfile::DigestCandidates(const MaxHeap& heap) {
    candidate_count = heap.size();
    if (!diagnose) {
        return;
    }
    auto copy = heap;
    while (!copy.empty()) {
        uint32_t bits;
        const auto candidate = copy.top();
        std::memcpy(&bits, &candidate.first, sizeof(bits));
        hash_word(candidate_digest, bits);
        hash_word(candidate_digest, candidate.second);
        copy.pop();
    }
}

void
SindiV2SearchProfile::DigestResults(const DatasetPtr& result) {
    if (!diagnose) {
        return;
    }
    hash_word(result_digest, static_cast<uint64_t>(result->GetDim()));
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        uint32_t bits;
        const auto distance = result->GetDistances()[i];
        std::memcpy(&bits, &distance, sizeof(bits));
        hash_word(result_digest, bits);
        hash_word(result_digest, static_cast<uint64_t>(result->GetIds()[i]));
    }
}

std::string
SindiV2SearchProfile::Dump(const SearchStatistics& statistics) const {
    auto json = statistics.ToJson();
    const auto us = [&](const char* key, uint64_t ns) {
        json[key].SetDouble(static_cast<double>(ns) / 1000.0);
    };
    json["sindi_v2_strategy"].SetString(parallel ? "ordered_batches" : "serial");
    json["sindi_v2_fallback"].SetString(fallback);
    json["sindi_v2_stop_reason"].SetString(stop);
    json["sindi_v2_execution_complete"].SetBool(std::string(stop) == "complete");
    json["sindi_v2_requested_slots"].SetUint64(requested);
    json["sindi_v2_granted_slots"].SetUint64(granted);
    json["sindi_v2_batches"].SetUint64(batches);
    json["sindi_v2_windows_computed"].SetUint64(computed_windows);
    json["sindi_v2_windows_consumed"].SetUint64(consumed_windows);
    json["sindi_v2_candidates"].SetUint64(candidate_count);
    json["sindi_v2_profile_enabled"].SetBool(enabled);
    json["sindi_v2_cpu_profile_enabled"].SetBool(profile_cpu);
    if (profile_cpu) {
        const bool compute_available = compute_cpu_complete && computed_windows > 0;
        json["sindi_v2_caller_cpu_available"].SetBool(caller_cpu.available);
        json["sindi_v2_compute_cpu_available"].SetBool(compute_available);
        if (caller_cpu.available) {
            us("sindi_v2_caller_cpu_us", caller_cpu.nanoseconds);
        }
        if (compute_available) {
            us("sindi_v2_compute_cpu_us", compute_cpu_ns);
        }
    }
    if (enabled) {
        us("sindi_v2_prepare_us", prepare_ns);
        us("sindi_v2_batch_wall_us", batch_ns);
        us("sindi_v2_caller_compute_us", caller_compute_ns);
        us("sindi_v2_join_wait_us", wait_ns);
        us("sindi_v2_compute_sum_us", compute_sum_ns);
        us("sindi_v2_compute_max_us", compute_max_ns);
        constexpr std::array<const char*, 6> buckets{
            "lt_50us", "50_200us", "200_1000us", "1000_5000us", "5000_20000us", "ge_20000us"};
        for (uint64_t i = 0; i < buckets.size(); ++i) {
            json[std::string("sindi_v2_window_compute_") + buckets[i]].SetUint64(
                compute_histogram[i]);
        }
        json["sindi_v2_slowest_window"].SetUint64(slowest_window);
        if (scan_postings_available) {
            json["sindi_v2_slowest_window_postings"].SetUint64(slowest_window_postings);
        }
        us("sindi_v2_slowest_batch_us", slowest_batch_ns);
        us("sindi_v2_slowest_batch_wait_us", slowest_batch_wait_ns);
        json["sindi_v2_slowest_batch_first_window"].SetUint64(slowest_batch_first);
        json["sindi_v2_slowest_batch_last_window"].SetUint64(slowest_batch_last);
        json["sindi_v2_slowest_batch_width"].SetUint64(slowest_batch_width);
        us("sindi_v2_consume_us", consume_ns);
        us("sindi_v2_rerank_us", rerank_ns);
        us("sindi_v2_output_us", output_ns);
        const auto total = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
        us("sindi_v2_total_us", total);
        const auto accounted = prepare_ns + batch_ns + consume_ns + rerank_ns + output_ns;
        us("sindi_v2_other_us", total > accounted ? total - accounted : 0);
        us("sindi_v2_rerank_read_us", rerank.read_ns);
        us("sindi_v2_rerank_compute_us", rerank.compute_ns);
        json["sindi_v2_rerank_logical_bytes"].SetUint64(rerank.read_bytes);
        json["sindi_v2_rerank_read_ranges"].SetUint64(rerank.read_ranges);
        json["sindi_v2_rerank_io_strategy"].SetString(rerank.io_strategy);
        json["sindi_v2_query_terms"].SetUint64(query_terms);
        if (scan_postings_available) {
            json["sindi_v2_scanned_postings"].SetUint64(scan_postings);
        }
    }
    if (diagnose) {
        json["sindi_v2_window_digest"].SetString(std::to_string(window_digest));
        json["sindi_v2_candidate_digest"].SetString(std::to_string(candidate_digest));
        json["sindi_v2_result_digest"].SetString(std::to_string(result_digest));
    }
    return json.Dump();
}

SindiV2WindowSlot::SindiV2WindowSlot(const SparseTermComputerPtr& query, uint32_t window_size)
    : allocator(SafeAllocator::FactoryDefaultAllocator()),
      computer(std::make_shared<SparseTermComputer>(*query, allocator.get())),
      dists(window_size, 0.0F, allocator.get()),
      context(allocator.get()) {
}

void
search_sindi_v2_batches(SearchExecutionContext& resources,
                        const SindiSearchTermDataCellPtr& datacell,
                        const SparseTermComputerPtr& computer,
                        uint32_t window_size,
                        uint32_t slot_limit,
                        bool use_term_lists,
                        const std::function<bool(uint32_t&, bool&)>& next,
                        const std::function<bool(SindiV2WindowSlot&)>& consume,
                        SindiV2SearchProfile& profile,
                        SearchStatistics* statistics) {
    const bool valid_slot_limit = slot_limit >= 1 && slot_limit <= profile.requested;
    CHECK_ARGUMENT(valid_slot_limit, "SINDI_V2 slot limit must be in [1, requested slots]");
    SearchExecutionReservation reservation(resources);
    if (resources.pool && !resources.pool->RequiresInlineExecution()) {
        reservation.extra = resources.TryReserve(slot_limit - 1);
    }
    profile.granted = reservation.extra + 1;
    std::vector<std::unique_ptr<SindiV2WindowSlot>> slots;
    slots.reserve(profile.granted);
    for (uint32_t i = 0; i < profile.granted; ++i) {
        slots.emplace_back(std::make_unique<SindiV2WindowSlot>(computer, window_size));
    }
    const auto compute = [&](SindiV2WindowSlot& slot) {
        slot.started = true;
        QueryProfileTimer timer(profile.enabled ? &slot.compute_ns : nullptr);
        SindiCpuTimer cpu_timer(profile.profile_cpu, slot.cpu);
        try {
            profile.CheckDeadline();
            slot.computer->SetTermPruneEnabled(slot.prune);
            datacell->QueryWindow(
                slot.dists.data(), slot.window, slot.computer, use_term_lists, slot.context);
        } catch (...) {
            slot.error = std::current_exception();
        }
    };
    for (;;) {
        profile.CheckDeadline();
        uint32_t count = 0;
        for (auto& slot : slots) {
            if (!next(slot->window, slot->prune)) {
                break;
            }
            std::fill(slot->dists.begin(), slot->dists.end(), 0.0F);
            slot->context.has_untracked_approximate_evaluations = false;
            slot->context.profile_work = profile.enabled;
            slot->context.scanned_postings = 0;
            slot->error = nullptr;
            slot->started = false;
            slot->compute_ns = 0;
            slot->cpu = {};
            ++count;
        }
        if (count == 0) {
            break;
        }
        ++profile.batches;
        std::vector<std::future<void>> futures;
        futures.reserve(count - 1);
        std::exception_ptr submit_error;
        uint64_t batch_ns = 0;
        uint64_t wait_ns = 0;
        {
            QueryProfileTimer batch(profile.enabled ? &batch_ns : nullptr);
            try {
                for (uint32_t i = 1; i < count; ++i) {
                    futures.emplace_back(
                        resources.pool->GeneralEnqueue([&, i] { compute(*slots[i]); }));
                }
                QueryProfileTimer caller(profile.enabled ? &profile.caller_compute_ns : nullptr);
                compute(*slots[0]);
            } catch (...) {
                submit_error = std::current_exception();
            }
            QueryProfileTimer wait(profile.enabled ? &wait_ns : nullptr);
            for (auto& future : futures) {
                try {
                    future.get();
                } catch (...) {
                    if (!submit_error) {
                        submit_error = std::current_exception();
                    }
                }
            }
        }
        profile.batch_ns += batch_ns;
        profile.wait_ns += wait_ns;
        if (profile.enabled && batch_ns >= profile.slowest_batch_ns) {
            profile.slowest_batch_ns = batch_ns;
            profile.slowest_batch_wait_ns = wait_ns;
            profile.slowest_batch_first = slots[0]->window;
            profile.slowest_batch_last = slots[count - 1]->window;
            profile.slowest_batch_width = count;
        }
        // Preserve completed work in failure diagnostics too. Tasks never submitted
        // after an enqueue error must not count as zero-cost completed windows.
        for (uint32_t i = 0; i < count; ++i) {
            auto& slot = *slots[i];
            if (!slot.started) {
                continue;
            }
            if (!slot.error) {
                ++profile.computed_windows;
            }
            profile.scan_postings += slot.context.scanned_postings;
            profile.compute_cpu_ns += slot.cpu.nanoseconds;
            profile.compute_cpu_complete = profile.compute_cpu_complete && slot.cpu.available;
            profile.RecordCompute(slot.window, slot.compute_ns, slot.context.scanned_postings);
            if (statistics != nullptr && slot.context.has_untracked_approximate_evaluations) {
                statistics->complete.store(false, std::memory_order_relaxed);
            }
        }
        // All tasks have drained before either infrastructure or window errors escape.
        if (submit_error) {
            std::rethrow_exception(submit_error);
        }
        profile.CheckDeadline();
        for (uint32_t i = 0; i < count; ++i) {
            profile.CheckDeadline();
            auto& slot = *slots[i];
            if (slot.error) {
                std::rethrow_exception(slot.error);
            }
            if (!consume(slot)) {
                return;
            }
        }
    }
}

}  // namespace vsag
