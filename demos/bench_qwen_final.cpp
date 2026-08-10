#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "base/allocator.h"
#include "base/config.h"
#include "model/causal_lm_model.h"
#include "op/math_ops.h"
#include "qwen_tokenizer.h"
#include "qwen_weight_loader.h"
#include "tensor/tensor.h"

using namespace hxinfer;

namespace {

constexpr int kInputLength = 1024;
constexpr int kOutputLength = 1024;
constexpr int kExpectedVocabSize = 152064;
constexpr int kConfiguredSequenceCapacity = 4096;
constexpr std::array<int, 3> kChatMlPrefix = {151644, 8948, 198};
constexpr std::array<int, 5> kChatMlSuffix = {151645, 198, 151644, 77091, 198};
constexpr std::array<int, 16> kHfOraclePrefix = {
    2132, 5868, 1075, 498, 3003, 3897, 1378, 19516,
    10010, 315, 1467, 11, 892, 4994, 311, 387,
};

void check_cuda(cudaError_t error, const std::string& operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(operation + ": " + cudaGetErrorString(error));
    }
}

std::string trim(const std::string& value) {
    const size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const size_t last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string token_fixture_path() {
    if (const char* configured = std::getenv("HXINFER_TOKEN_IDS_FILE")) {
        if (*configured != '\0') {
            return configured;
        }
    }

    const std::array<std::string, 3> candidates = {
        "tests/fixtures/qwen_sharegpt_1024_token_ids.txt",
        "/workspace/project/hxinfer/tests/fixtures/qwen_sharegpt_1024_token_ids.txt",
        "/workspace/qwen_sharegpt_1024_token_ids.txt",
    };
    for (const auto& candidate : candidates) {
        std::ifstream probe(candidate);
        if (probe.good()) {
            return candidate;
        }
    }
    throw std::runtime_error(
        "Qwen ShareGPT token fixture not found; set HXINFER_TOKEN_IDS_FILE or run from the repository root");
}

std::vector<int> load_token_ids(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open token fixture: " + path);
    }

    std::ostringstream contents;
    contents << input.rdbuf();
    const std::string encoded = trim(contents.str());
    if (encoded.empty() || encoded.back() == ',') {
        throw std::runtime_error("Malformed token fixture: empty content or trailing comma");
    }

    std::vector<int> tokens;
    std::istringstream fields(encoded);
    std::string field;
    while (std::getline(fields, field, ',')) {
        field = trim(field);
        if (field.empty()) {
            throw std::runtime_error("Malformed token fixture: empty token field");
        }
        size_t parsed = 0;
        long long token = 0;
        try {
            token = std::stoll(field, &parsed, 10);
        } catch (const std::exception&) {
            throw std::runtime_error("Malformed token ID: " + field);
        }
        if (parsed != field.size()) {
            throw std::runtime_error("Malformed token ID: " + field);
        }
        if (token < 0 || token >= kExpectedVocabSize) {
            throw std::runtime_error("Out-of-range token ID: " + std::to_string(token));
        }
        tokens.push_back(static_cast<int>(token));
    }

    if (tokens.size() != kInputLength) {
        throw std::runtime_error(
            "Expected exactly 1024 input token IDs, got " + std::to_string(tokens.size()));
    }
    for (size_t i = 0; i < kChatMlPrefix.size(); ++i) {
        if (tokens[i] != kChatMlPrefix[i]) {
            throw std::runtime_error("Token fixture does not have the expected Qwen ChatML system prefix");
        }
    }
    for (size_t i = 0; i < kChatMlSuffix.size(); ++i) {
        if (tokens[tokens.size() - kChatMlSuffix.size() + i] != kChatMlSuffix[i]) {
            throw std::runtime_error("Token fixture does not end with the Qwen ChatML assistant generation prompt");
        }
    }
    return tokens;
}

void require_finite_logits(const std::shared_ptr<Tensor>& logits, int count, const std::string& phase) {
    std::vector<__half> host_logits(count);
    check_cuda(
        cudaMemcpy(host_logits.data(), logits->raw_data_ptr(), count * sizeof(__half), cudaMemcpyDeviceToHost),
        "Copy " + phase + " logits to host");
    for (int i = 0; i < count; ++i) {
        if (!std::isfinite(__half2float(host_logits[i]))) {
            throw std::runtime_error(phase + " logits contain NaN/Inf at vocabulary index " + std::to_string(i));
        }
    }
}

size_t oracle_prefix_length(const std::vector<int>& generated) {
    size_t matched = 0;
    while (matched < generated.size() && matched < kHfOraclePrefix.size() &&
           generated[matched] == kHfOraclePrefix[matched]) {
        ++matched;
    }
    return matched;
}

}  // namespace

int main() {
    try {
        const char* env_data = std::getenv("HXINFER_DATA_DIR");
        const std::string data_dir =
            env_data ? std::string(env_data) : "/workspace/models/Qwen2.5-7B-Instruct";
        const std::string fixture_path = token_fixture_path();

        if (kInputLength + kOutputLength > kConfiguredSequenceCapacity) {
            throw std::runtime_error("Requested input + output exceeds the configured 4096-token KV capacity");
        }
        const std::vector<int> tokens = load_token_ids(fixture_path);

        std::cout << "Fixture: " << fixture_path << '\n';
        std::cout << "Input: " << tokens.size() << " fixed ShareGPT ChatML token IDs\n";
        std::cout << "Output: " << kOutputLength << " greedy tokens (ignore EOS)\n";
        std::cout << "Model: " << data_dir << '\n';

        auto cpu_alloc = std::make_shared<CPUAllocator>();
        auto cuda_alloc = std::make_shared<CUDAAllocator>();
        ModelConfig config;

        std::cout << ">>> Loading Qwen2.5-7B-Instruct...\n";
        auto model = QwenWeightLoader::load(data_dir, config, cpu_alloc, cuda_alloc);
        if (config.vocab_size != kExpectedVocabSize) {
            throw std::runtime_error(
                "Unexpected Qwen vocabulary size: " + std::to_string(config.vocab_size));
        }
        if (kInputLength + kOutputLength > config.seq_len) {
            throw std::runtime_error(
                "Requested 1024 input + 1024 output exceeds model sequence capacity " +
                std::to_string(config.seq_len));
        }

        std::cout << ">>> Loading tokenizer for output decoding...\n";
        QwenTokenizer tokenizer(data_dir + "/tokenizer.json");

        auto input_ids_cpu = std::make_shared<Tensor>(
            cpu_alloc, std::vector<int>{kInputLength}, DataType::kDataTypeFP32);
        auto input_ids_gpu = std::make_shared<Tensor>(
            cuda_alloc, std::vector<int>{kInputLength}, DataType::kDataTypeFP32);
        input_ids_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
        int* ids_ptr = input_ids_cpu->tensor_data_ptr<int>();
        for (int i = 0; i < kInputLength; ++i) {
            ids_ptr[i] = tokens[i];
        }
        check_cuda(
            cudaMemcpy(
                input_ids_gpu->raw_data_ptr(), input_ids_cpu->raw_data_ptr(),
                kInputLength * sizeof(int), cudaMemcpyHostToDevice),
            "Copy input IDs to GPU");

        auto logits_prefill = std::make_shared<Tensor>(
            cuda_alloc, std::vector<int>{kInputLength, config.vocab_size}, DataType::kDataTypeFP16);
        logits_prefill->tensor_set_device_type(DeviceType::kDeviceCUDA);

        auto input_single_cpu = std::make_shared<Tensor>(
            cpu_alloc, std::vector<int>{1}, DataType::kDataTypeFP32);
        auto input_single_gpu = std::make_shared<Tensor>(
            cuda_alloc, std::vector<int>{1}, DataType::kDataTypeFP32);
        auto logits_decode = std::make_shared<Tensor>(
            cuda_alloc, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP16);
        input_single_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
        logits_decode->tensor_set_device_type(DeviceType::kDeviceCUDA);

        const auto prefill_start = std::chrono::high_resolution_clock::now();
        model->forward_prefill(input_ids_gpu, logits_prefill, kInputLength);
        check_cuda(cudaDeviceSynchronize(), "Synchronize prefill");
        const auto prefill_end = std::chrono::high_resolution_clock::now();
        const double prefill_ms =
            std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();

        auto last_row = std::make_shared<Tensor>(
            cuda_alloc, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP16);
        last_row->tensor_set_device_type(DeviceType::kDeviceCUDA);
        check_cuda(
            cudaMemcpy(
                last_row->raw_data_ptr(),
                static_cast<char*>(logits_prefill->raw_data_ptr()) +
                    (kInputLength - 1) * config.vocab_size * sizeof(__half),
                config.vocab_size * sizeof(__half), cudaMemcpyDeviceToDevice),
            "Copy final prefill logit row");
        require_finite_logits(last_row, config.vocab_size, "prefill");

        int next_token = argmax_tensor(last_row);
        int position = kInputLength;
        std::vector<int> generated_ids;
        generated_ids.reserve(kOutputLength);
        std::string output_text;

        const auto decode_start = std::chrono::high_resolution_clock::now();
        while (static_cast<int>(generated_ids.size()) < kOutputLength) {
            generated_ids.push_back(next_token);
            output_text += tokenizer.decode(next_token);
            if (static_cast<int>(generated_ids.size()) == kOutputLength) {
                break;
            }

            input_single_cpu->tensor_data_ptr<int>()[0] = next_token;
            check_cuda(
                cudaMemcpy(
                    input_single_gpu->raw_data_ptr(), input_single_cpu->raw_data_ptr(),
                    sizeof(int), cudaMemcpyHostToDevice),
                "Copy decode input ID to GPU");
            model->forward(input_single_gpu, logits_decode, position);
            check_cuda(cudaDeviceSynchronize(), "Synchronize decode position " + std::to_string(position));
            require_finite_logits(
                logits_decode, config.vocab_size,
                "decode position " + std::to_string(position));
            next_token = argmax_tensor(logits_decode);
            ++position;
        }
        check_cuda(cudaDeviceSynchronize(), "Synchronize completed generation");
        const auto decode_end = std::chrono::high_resolution_clock::now();

        const double decode_ms =
            std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
        const double total_ms = prefill_ms + decode_ms;
        const std::unordered_set<int> unique_tokens(generated_ids.begin(), generated_ids.end());
        const size_t oracle_match = oracle_prefix_length(generated_ids);
        const bool repeated_o_prefix =
            output_text.size() >= 32 && output_text.compare(0, 32, std::string(32, 'O')) == 0;

        std::cout << "\n--- Generated Token IDs (1024) ---\n";
        for (size_t i = 0; i < generated_ids.size(); ++i) {
            if (i != 0) {
                std::cout << ',';
            }
            std::cout << generated_ids[i];
        }
        std::cout << "\n--- Generated Text (full) ---\n";
        std::cout << output_text << '\n';
        std::cout << "--- End Generated Text ---\n";

        std::cout << "\n============ Result ============\n";
        std::cout << "Input tokens: " << tokens.size() << '\n';
        std::cout << "Output tokens: " << generated_ids.size() << '\n';
        std::cout << "Unique output tokens: " << unique_tokens.size() << '\n';
        std::cout << "Finite logits: yes (prefill and every decode step)\n";
        std::cout << "HF oracle prefix match: " << oracle_match << '/' << kHfOraclePrefix.size() << '\n';
        std::cout << "Prefill: " << prefill_ms / 1000.0 << " s, "
                  << kInputLength / (prefill_ms / 1000.0) << " tok/s\n";
        std::cout << "TTFT: " << prefill_ms << " ms\n";
        std::cout << "Decode: " << decode_ms / 1000.0 << " s for "
                  << (kOutputLength - 1) << " post-prefill forward steps\n";
        std::cout << "ITL: " << decode_ms / (kOutputLength - 1) << " ms\n";
        std::cout << "Total: " << total_ms / 1000.0 << " s\n";
        std::cout << "================================\n";

        if (generated_ids.front() != kHfOraclePrefix.front()) {
            std::cerr << "QUALITY FAILURE: first greedy token " << generated_ids.front()
                      << " differs from HuggingFace oracle " << kHfOraclePrefix.front() << '\n';
            return 2;
        }
        if (unique_tokens.size() <= 1) {
            std::cerr << "QUALITY FAILURE: output collapsed to one token\n";
            return 3;
        }
        if (repeated_o_prefix) {
            std::cerr << "QUALITY FAILURE: output begins with 32 repeated 'O' characters\n";
            return 4;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
