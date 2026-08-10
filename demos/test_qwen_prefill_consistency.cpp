#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "base/allocator.h"
#include "base/config.h"
#include "model/causal_lm_model.h"
#include "qwen_weight_loader.h"
#include "tensor/tensor.h"

using namespace hxinfer;

namespace {

constexpr int kTopK = 10;

void check_cuda(cudaError_t error, const std::string& operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(operation + ": " + cudaGetErrorString(error));
    }
}

std::vector<int> load_prefix(const std::string& path, int sequence_length) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open token fixture: " + path);
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    std::istringstream fields(contents.str());
    std::vector<int> tokens;
    std::string field;
    while (std::getline(fields, field, ',') && static_cast<int>(tokens.size()) < sequence_length) {
        size_t parsed = 0;
        const int token = std::stoi(field, &parsed, 10);
        while (parsed < field.size() && std::isspace(static_cast<unsigned char>(field[parsed]))) {
            ++parsed;
        }
        if (parsed != field.size()) {
            throw std::runtime_error("Malformed fixture token: " + field);
        }
        tokens.push_back(token);
    }
    if (static_cast<int>(tokens.size()) != sequence_length) {
        throw std::runtime_error("Fixture is shorter than requested sequence length");
    }
    if (tokens.size() < 3 || tokens[0] != 151644 || tokens[1] != 8948 || tokens[2] != 198) {
        throw std::runtime_error("Fixture is not the official Qwen ChatML sequence");
    }
    return tokens;
}

std::vector<float> copy_logits(const std::shared_ptr<Tensor>& logits, int vocab_size) {
    std::vector<__half> half_logits(vocab_size);
    check_cuda(
        cudaMemcpy(half_logits.data(), logits->raw_data_ptr(), vocab_size * sizeof(__half), cudaMemcpyDeviceToHost),
        "Copy logits to host");
    std::vector<float> result(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        result[i] = __half2float(half_logits[i]);
        if (!std::isfinite(result[i])) {
            throw std::runtime_error("Non-finite logit at index " + std::to_string(i));
        }
    }
    return result;
}

std::vector<float> run_serial(
    const std::string& model_dir, const std::vector<int>& tokens, int& vocab_size) {
    auto cpu = std::make_shared<CPUAllocator>();
    auto cuda = std::make_shared<CUDAAllocator>();
    ModelConfig config;
    auto model = QwenWeightLoader::load(model_dir, config, cpu, cuda);
    vocab_size = config.vocab_size;

    auto input_cpu = std::make_shared<Tensor>(cpu, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto input_gpu = std::make_shared<Tensor>(cuda, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto logits = std::make_shared<Tensor>(
        cuda, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP16);
    input_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
    logits->tensor_set_device_type(DeviceType::kDeviceCUDA);

    for (size_t position = 0; position < tokens.size(); ++position) {
        input_cpu->tensor_data_ptr<int>()[0] = tokens[position];
        check_cuda(
            cudaMemcpy(input_gpu->raw_data_ptr(), input_cpu->raw_data_ptr(), sizeof(int), cudaMemcpyHostToDevice),
            "Copy serial token");
        model->forward(input_gpu, logits, static_cast<int>(position));
    }
    check_cuda(cudaDeviceSynchronize(), "Synchronize serial prefill");
    return copy_logits(logits, config.vocab_size);
}

std::vector<float> run_batch(
    const std::string& model_dir, const std::vector<int>& tokens, int expected_vocab_size) {
    auto cpu = std::make_shared<CPUAllocator>();
    auto cuda = std::make_shared<CUDAAllocator>();
    ModelConfig config;
    auto model = QwenWeightLoader::load(model_dir, config, cpu, cuda);
    if (config.vocab_size != expected_vocab_size) {
        throw std::runtime_error("Fresh-model vocabulary sizes differ");
    }

    const int sequence_length = static_cast<int>(tokens.size());
    auto input_cpu = std::make_shared<Tensor>(
        cpu, std::vector<int>{sequence_length}, DataType::kDataTypeFP32);
    auto input_gpu = std::make_shared<Tensor>(
        cuda, std::vector<int>{sequence_length}, DataType::kDataTypeFP32);
    auto logits = std::make_shared<Tensor>(
        cuda, std::vector<int>{sequence_length, config.vocab_size}, DataType::kDataTypeFP16);
    auto last_row = std::make_shared<Tensor>(
        cuda, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP16);
    input_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
    logits->tensor_set_device_type(DeviceType::kDeviceCUDA);
    last_row->tensor_set_device_type(DeviceType::kDeviceCUDA);

    std::copy(tokens.begin(), tokens.end(), input_cpu->tensor_data_ptr<int>());
    check_cuda(
        cudaMemcpy(
            input_gpu->raw_data_ptr(), input_cpu->raw_data_ptr(),
            sequence_length * sizeof(int), cudaMemcpyHostToDevice),
        "Copy batch tokens");
    model->forward_prefill(input_gpu, logits, sequence_length);
    check_cuda(cudaDeviceSynchronize(), "Synchronize batch prefill");
    check_cuda(
        cudaMemcpy(
            last_row->raw_data_ptr(),
            static_cast<char*>(logits->raw_data_ptr()) +
                static_cast<size_t>(sequence_length - 1) * config.vocab_size * sizeof(__half),
            config.vocab_size * sizeof(__half), cudaMemcpyDeviceToDevice),
        "Copy batch final row");
    return copy_logits(last_row, config.vocab_size);
}

std::vector<std::pair<int, float>> top_k(const std::vector<float>& logits) {
    std::vector<std::pair<int, float>> indexed;
    indexed.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        indexed.emplace_back(static_cast<int>(i), logits[i]);
    }
    std::partial_sort(
        indexed.begin(), indexed.begin() + kTopK, indexed.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });
    indexed.resize(kTopK);
    return indexed;
}

void print_top_k(const char* label, const std::vector<std::pair<int, float>>& values) {
    std::cout << label << ':';
    for (const auto& [token, logit] : values) {
        std::cout << ' ' << token << '(' << logit << ')';
    }
    std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const int sequence_length = argc > 1 ? std::stoi(argv[1]) : 32;
        if (sequence_length < 3 || sequence_length > 1024) {
            throw std::runtime_error("Sequence length must be between 3 and 1024");
        }
        const char* env_model = std::getenv("HXINFER_DATA_DIR");
        const char* env_fixture = std::getenv("HXINFER_TOKEN_IDS_FILE");
        const std::string model_dir =
            env_model ? env_model : "/workspace/models/Qwen2.5-7B-Instruct";
        const std::string fixture =
            env_fixture ? env_fixture : "tests/fixtures/qwen_sharegpt_1024_token_ids.txt";
        const std::vector<int> tokens = load_prefix(fixture, sequence_length);

        int vocab_size = 0;
        std::cout << "Running fresh-model serial prefill, S=" << sequence_length << "\n";
        const std::vector<float> serial = run_serial(model_dir, tokens, vocab_size);
        check_cuda(cudaDeviceSynchronize(), "Synchronize before fresh batch model");
        std::cout << "Running fresh-model batch prefill, S=" << sequence_length << "\n";
        const std::vector<float> batch = run_batch(model_dir, tokens, vocab_size);

        const auto serial_top = top_k(serial);
        const auto batch_top = top_k(batch);
        double mean_absolute_error = 0.0;
        float max_absolute_error = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            const float error = std::fabs(serial[i] - batch[i]);
            mean_absolute_error += error;
            max_absolute_error = std::max(max_absolute_error, error);
        }
        mean_absolute_error /= vocab_size;

        print_top_k("serial top-10", serial_top);
        print_top_k("batch top-10", batch_top);
        std::cout << "logit MAE=" << mean_absolute_error
                  << " max_abs=" << max_absolute_error << '\n';
        if (serial_top.front().first != batch_top.front().first) {
            std::cerr << "MISMATCH: serial top-1=" << serial_top.front().first
                      << " batch top-1=" << batch_top.front().first << '\n';
            return 2;
        }
        std::cout << "MATCH: fresh-model serial and batch top-1="
                  << serial_top.front().first << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
