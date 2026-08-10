#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "qwen_quality_gate.h"

namespace {

constexpr std::array<int, 16> kCorrectPrefix = {
    2132, 5868, 1075, 498, 3003, 3897, 1378, 19516,
    10010, 315, 1467, 11, 892, 4994, 311, 387,
};

std::vector<int> make_noncollapsed_sequence() {
    std::vector<int> tokens(kCorrectPrefix.begin(), kCorrectPrefix.end());
    for (int i = 0; i < 128; ++i) {
        tokens.push_back(1000 + (i % 7));
    }
    return tokens;
}

}  // namespace

int main() {
    try {
        const std::vector<int> bf16_reference(kCorrectPrefix.begin(), kCorrectPrefix.end());
        const std::vector<int> fp16_reference(kCorrectPrefix.begin(), kCorrectPrefix.end());

        const std::vector<int> healthy = make_noncollapsed_sequence();
        const auto healthy_metrics = hxinfer::qwen_benchmark::evaluate_quality(
            healthy, bf16_reference, fp16_reference);
        if (!healthy_metrics.accepted() || healthy_metrics.max_identical_token_run != 1) {
            throw std::runtime_error("healthy sequence was rejected by the quality gate");
        }

        std::vector<int> collapsed(kCorrectPrefix.begin(), kCorrectPrefix.end());
        collapsed.insert(collapsed.end(), 700, 15);
        const auto collapsed_metrics = hxinfer::qwen_benchmark::evaluate_quality(
            collapsed, bf16_reference, fp16_reference);
        if (collapsed_metrics.accepted() || collapsed_metrics.bf16_prefix_length != 16 ||
            collapsed_metrics.max_identical_token_run != 700) {
            throw std::runtime_error("post-prefix token-15 collapse was not rejected");
        }

        std::vector<int> at_run_limit(kCorrectPrefix.begin(), kCorrectPrefix.end());
        at_run_limit.insert(at_run_limit.end(), 32, 15);
        if (!hxinfer::qwen_benchmark::evaluate_quality(
                 at_run_limit, bf16_reference, fp16_reference).accepted()) {
            throw std::runtime_error("32-token run at the documented limit was rejected");
        }
        at_run_limit.push_back(15);
        if (hxinfer::qwen_benchmark::evaluate_quality(
                at_run_limit, bf16_reference, fp16_reference).accepted()) {
            throw std::runtime_error("33-token run above the documented limit was accepted");
        }

        std::vector<int> wrong_prefix = healthy;
        wrong_prefix[5] = 999;
        const auto wrong_prefix_metrics = hxinfer::qwen_benchmark::evaluate_quality(
            wrong_prefix, bf16_reference, fp16_reference);
        if (wrong_prefix_metrics.accepted()) {
            throw std::runtime_error("sequence with a wrong BF16 prefix was not rejected");
        }

        std::cout << "PASS: healthy/32-run accepted; 33-run, 700-collapse, and wrong prefix rejected\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
