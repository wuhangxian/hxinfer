#ifndef HXINFER_QWEN_QUALITY_GATE_H
#define HXINFER_QWEN_QUALITY_GATE_H

#include <algorithm>
#include <cstddef>
#include <vector>

namespace hxinfer::qwen_benchmark {

constexpr std::size_t kRequiredBf16Prefix = 16;
constexpr std::size_t kMaxIdenticalTokenRun = 32;

struct QualityMetrics {
    std::size_t bf16_prefix_length = 0;
    std::size_t fp16_prefix_length = 0;
    std::size_t max_identical_token_run = 0;

    bool accepted() const {
        return bf16_prefix_length >= kRequiredBf16Prefix &&
               max_identical_token_run <= kMaxIdenticalTokenRun;
    }
};

inline std::size_t longest_common_prefix(
    const std::vector<int>& generated,
    const std::vector<int>& reference) {
    const std::size_t limit = std::min(generated.size(), reference.size());
    std::size_t matched = 0;
    while (matched < limit && generated[matched] == reference[matched]) {
        ++matched;
    }
    return matched;
}

inline std::size_t max_consecutive_identical_run(const std::vector<int>& tokens) {
    if (tokens.empty()) {
        return 0;
    }
    std::size_t maximum = 1;
    std::size_t current = 1;
    for (std::size_t i = 1; i < tokens.size(); ++i) {
        if (tokens[i] == tokens[i - 1]) {
            ++current;
            maximum = std::max(maximum, current);
        } else {
            current = 1;
        }
    }
    return maximum;
}

inline QualityMetrics evaluate_quality(
    const std::vector<int>& generated,
    const std::vector<int>& bf16_reference,
    const std::vector<int>& fp16_reference) {
    return {
        longest_common_prefix(generated, bf16_reference),
        longest_common_prefix(generated, fp16_reference),
        max_consecutive_identical_run(generated),
    };
}

}  // namespace hxinfer::qwen_benchmark

#endif  // HXINFER_QWEN_QUALITY_GATE_H
