// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include <fmt/format.h>

#include <array>
#include <future>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "algorithm/sindi_v2/sindi_v2.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/searcher/search_execution_context.h"
#include "io/memory_block_io/memory_block_io_parameter.h"
#include "io/memory_io/memory_io_parameter.h"
#include "unittest.h"

using namespace vsag;

namespace {
class CallerFilter final : public Filter {
public:
    explicit CallerFilter(bool fail = false) : owner(std::this_thread::get_id()), fail(fail) {
    }
    bool
    CheckValid(int64_t id) const override {
        if (std::this_thread::get_id() != owner) {
            throw std::logic_error("filter left its original calling thread");
        }
        if (fail)
            throw std::runtime_error("injected filter failure");
        calls.push_back(id);
        return id % 3 != 0;
    }
    std::thread::id owner;
    bool fail;
    mutable std::vector<int64_t> calls;
};

void
SameResults(const DatasetPtr& actual, const DatasetPtr& expected) {
    REQUIRE(actual->GetDim() == expected->GetDim());
    for (int64_t i = 0; i < actual->GetDim(); ++i) {
        REQUIRE(actual->GetIds()[i] == expected->GetIds()[i]);
        REQUIRE(actual->GetDistances()[i] == expected->GetDistances()[i]);
    }
}
}  // namespace

TEST_CASE("SINDIV2 batches preserve global candidates and serial filter order",
          "[ut][SINDIV2][SINDIV2Parallel]") {
    const auto quantization = GENERATE(SparseValueQuantizationType::FP32,
                                       SparseValueQuantizationType::FP16,
                                       SparseValueQuantizationType::SQ8);
    const bool reorder = GENERATE(false, true);
    const bool pruning = GENERATE(false, true);
    const uint32_t workers = GENERATE(0U, 3U);
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.allocator_ = allocator;
    common.metric_ = MetricType::METRIC_TYPE_IP;
    common.dim_ = 3;
    common.search_resources_ = std::make_shared<SearchExecutionContext>(workers);
    auto parameters = std::make_shared<SINDIV2Parameter>();
    parameters->term_id_limit = 8;
    parameters->window_size = 5;
    parameters->immutable = true;
    parameters->doc_prune_ratio = pruning ? 0.5F : 0.0F;
    parameters->use_reorder = reorder;
    parameters->remap_term_ids = GENERATE(false, true);
    parameters->rerank_type = "fp16";
    parameters->use_quantization = quantization != SparseValueQuantizationType::FP32;
    parameters->sparse_value_quant_type = quantization;
    parameters->term_io_parameter = std::make_shared<MemoryIOParameter>();
    parameters->rerank_io_parameter = std::make_shared<MemoryBlockIOParameter>();
    SINDIV2 index(parameters, common);

    // Repeated scores, overlapping terms, and a partial final window exercise
    // serial tie eviction and the per-window mapped-term state.
    constexpr uint64_t count = 37;
    std::array<uint32_t, 3> terms{1, 2, 3};
    std::vector<std::array<float, 3>> weights(count);
    std::vector<SparseVector> vectors(count);
    std::vector<int64_t> labels(count);
    for (uint64_t i = 0; i < count; ++i) {
        weights[i] = {1.0F + static_cast<float>(i % 5), 0.2F, 0.1F};
        vectors[i] = SparseVector{3, terms.data(), weights[i].data()};
        labels[i] = static_cast<int64_t>(100 + i);
    }
    auto base = Dataset::Make();
    base->NumElements(count)->Ids(labels.data())->SparseVectors(vectors.data())->Owner(false);
    REQUIRE(index.Build(base).empty());
    std::array<uint32_t, 2> query_terms{1, 3};
    std::array<float, 2> query_weights{1.0F, 0.5F};
    SparseVector sparse{2, query_terms.data(), query_weights.data()};
    auto query = Dataset::Make();
    query->NumElements(1)->SparseVectors(&sparse)->Owner(false);
    const auto request = [pruning](uint32_t p, uint64_t budget) {
        return fmt::format(R"({{"sindi_v2":{{"window_parallelism":{},"n_candidate":9,
            "query_prune_ratio":{},"term_prune_ratio":{},"term_retain_threshold":{},
            "filter_callback_limit":{},"profile_search":true,"diagnose_search":true}}}})",
                           p,
                           pruning ? 0.3 : 0.0,
                           pruning ? 0.4 : 0.0,
                           pruning ? 8 : 0,
                           budget);
    };
    const std::vector<std::string> digests{"sindi_v2_window_digest", "sindi_v2_candidate_digest"};
    for (const auto budget : {uint64_t{0}, uint64_t{3}}) {
        auto serial_filter = std::make_shared<CallerFilter>();
        const auto serial = index.KnnSearch(query, 5, request(1, budget), serial_filter);
        for (const auto p : {2U, 4U, 4U}) {
            auto filter = std::make_shared<CallerFilter>();
            const auto parallel = index.KnnSearch(query, 5, request(p, budget), filter);
            SameResults(parallel, serial);
            REQUIRE(filter->calls == serial_filter->calls);
            REQUIRE(parallel->GetStatistics(digests) == serial->GetStatistics(digests));
            REQUIRE(parallel->GetStatistics({"sindi_v2_strategy"})[0] == "\"ordered_batches\"");
            if (reorder && parallel->GetDim() > 0) {
                const auto io = parallel->GetStatistics({"sindi_v2_rerank_logical_bytes",
                                                         "sindi_v2_rerank_read_ranges",
                                                         "sindi_v2_rerank_io_strategy"});
                REQUIRE(std::stoull(io[0]) > 0);
                REQUIRE(std::stoull(io[1]) > 0);
                REQUIRE_FALSE(io[2].empty());
                REQUIRE(io[2] != "\"unavailable\"");
            }
            REQUIRE(common.search_resources_->busy == 0);
            REQUIRE(common.search_resources_->callers == 0);
        }
    }
    SECTION("deadline and filter failure drain resources and leave the index usable") {
        REQUIRE_THROWS(index.KnnSearch(
            query, 5, R"({"sindi_v2":{"window_parallelism":4,"timeout_ms":0}})", nullptr));
        REQUIRE_THROWS(
            index.KnnSearch(query, 5, request(4, 0), std::make_shared<CallerFilter>(true)));
        REQUIRE(common.search_resources_->busy == 0);
        REQUIRE(common.search_resources_->callers == 0);
        SameResults(index.KnnSearch(query, 5, request(4, 0), nullptr),
                    index.KnnSearch(query, 5, request(1, 0), nullptr));
    }
    SECTION("SearchWithRequest shares candidate and threshold semantics") {
        SearchRequest search;
        search.query_ = query;
        search.topk_ = 5;
        search.params_str_ = request(4, 0);
        search.threshold_ = 0.0F;
        auto json = JsonType::Parse(request(1, 0));
        json["threshold"].SetFloat(0.0F);
        SameResults(index.SearchWithRequest(search),
                    index.KnnSearch(query, 5, json.Dump(), nullptr));
    }
    SECTION("missing resources retain the original serial path") {
        IndexCommonParam serial_common = common;
        serial_common.search_resources_.reset();
        SINDIV2 serial_index(parameters, serial_common);
        REQUIRE(serial_index.Build(base).empty());
        const auto actual = serial_index.KnnSearch(query, 5, request(4, 0), nullptr);
        SameResults(actual, serial_index.KnnSearch(query, 5, request(1, 0), nullptr));
        REQUIRE(actual->GetStatistics({"sindi_v2_fallback"})[0] == "\"no_resources\"");
    }
    SECTION("one window retains the parallel deadline with just the caller slot") {
        auto one_window = Dataset::Make()
                              ->NumElements(4)
                              ->Ids(labels.data())
                              ->SparseVectors(vectors.data())
                              ->Owner(false);
        SINDIV2 small_index(parameters, common);
        REQUIRE(small_index.Build(one_window).empty());
        const auto actual = small_index.KnnSearch(query, 5, request(4, 0), nullptr);
        SameResults(actual, small_index.KnnSearch(query, 5, request(1, 0), nullptr));
        REQUIRE(actual->GetStatistics({"sindi_v2_strategy"})[0] == "\"ordered_batches\"");
        REQUIRE(actual->GetStatistics({"sindi_v2_granted_slots"})[0] == "1");
        REQUIRE_THROWS(small_index.KnnSearch(
            query, 5, R"({"sindi_v2":{"window_parallelism":4,"timeout_ms":0}})", nullptr));
        REQUIRE(common.search_resources_->busy == 0);
        REQUIRE(common.search_resources_->callers == 0);
    }
    SECTION("range remains serial when parallelism is requested") {
        const auto actual = index.RangeSearch(query, 2.0F, request(4, 0), nullptr, 10);
        SameResults(actual, index.RangeSearch(query, 2.0F, request(1, 0), nullptr, 10));
        REQUIRE(actual->GetStatistics({"sindi_v2_fallback"})[0] == "\"unsupported_mode\"");
    }
    SECTION("metadata routing keeps the same window order and boundary pruning") {
        std::vector<std::string> hosts(count);
        std::vector<int64_t> timestamps(count);
        for (uint64_t i = 0; i < count; ++i) {
            hosts[i] = i % 2 == 0 ? "host-a" : "host-b";
            timestamps[i] = (10 + i % 3) * SINDI_SECONDS_PER_DAY + i;
        }
        base->StringMetadata(SINDI_HOST_METADATA_NAME, hosts.data())
            ->Int64Metadata(SINDI_PUBLISH_TIME_METADATA_NAME, timestamps.data());
        SINDIV2 metadata_index(parameters, common);
        REQUIRE(metadata_index.Build(base).empty());
        std::string host = "host-a";
        int64_t begin = 10 * SINDI_SECONDS_PER_DAY;
        int64_t end = 11 * SINDI_SECONDS_PER_DAY + 100;
        query->StringMetadata(SINDI_HOST_METADATA_NAME, &host)
            ->Int64Metadata(SINDI_PUBLISH_TIME_BEGIN_METADATA_NAME, &begin)
            ->Int64Metadata(SINDI_PUBLISH_TIME_END_METADATA_NAME, &end);
        auto serial_filter = std::make_shared<CallerFilter>();
        auto parallel_filter = std::make_shared<CallerFilter>();
        const auto expected = metadata_index.KnnSearch(query, 5, request(1, 0), serial_filter);
        const auto actual = metadata_index.KnnSearch(query, 5, request(4, 0), parallel_filter);
        SameResults(actual, expected);
        REQUIRE(parallel_filter->calls == serial_filter->calls);
        REQUIRE(actual->GetStatistics(digests) == expected->GetStatistics(digests));
    }
    SECTION("concurrent callers share admission without sharing mutable window state") {
        const auto expected = index.KnnSearch(query, 5, request(1, 0), nullptr);
        std::vector<std::future<DatasetPtr>> futures;
        for (uint32_t i = 0; i < 4; ++i) {
            futures.emplace_back(std::async(std::launch::async, [&] {
                return index.KnnSearch(query, 5, request(4, 0), nullptr);
            }));
        }
        for (auto& future : futures) {
            SameResults(future.get(), expected);
        }
        REQUIRE(common.search_resources_->busy == 0);
        REQUIRE(common.search_resources_->callers == 0);
    }
    SECTION("queries already on the shared pool compute inline without nested joins") {
        if (common.search_resources_->pool) {
            const auto expected = index.KnnSearch(query, 5, request(1, 0), nullptr);
            auto nested = common.search_resources_->pool->GeneralEnqueue(
                [&] { return index.KnnSearch(query, 5, request(4, 0), nullptr); });
            const auto actual = nested.get();
            SameResults(actual, expected);
            REQUIRE(actual->GetStatistics({"sindi_v2_granted_slots"})[0] == "1");
        }
    }
}

TEST_CASE("SINDIV2 parallel options reject candidate quotas and invalid controls",
          "[ut][SINDIV2][SINDIV2Parameter][SINDIV2Parallel]") {
    SINDIV2SearchParameter parameter;
    parameter.FromJson(JsonType::Parse(R"({"sindi_v2":{"window_parallelism":4,
        "profile_search":true,"diagnose_search":true}})"));
    REQUIRE(parameter.window_parallelism == 4);
    REQUIRE(parameter.profile_search);
    REQUIRE(parameter.diagnose_search);
    SINDIV2SearchParameter roundtrip;
    roundtrip.FromJson(parameter.ToJson());
    REQUIRE(roundtrip.window_parallelism == 4);
    REQUIRE(roundtrip.profile_search);
    for (const char* json : {R"({"sindi_v2":{"window_parallelism":0}})",
                             R"({"sindi_v2":{"window_parallelism":17}})",
                             R"({"sindi_v2":{"window_parallelism":2.5}})",
                             R"({"sindi_v2":{"parallel_n_candidate_ratio":2}})",
                             R"({"sindi_v2":{"profile_search":"true"}})"}) {
        REQUIRE_THROWS(parameter.FromJson(JsonType::Parse(json)));
    }
}
