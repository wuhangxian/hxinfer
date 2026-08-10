#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "base/allocator.h"
#include "base/config.h"
#include "op/math_ops.h"
#include "tensor/tensor.h"

using namespace hxinfer;

namespace {

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

std::vector<float> qwen_rope_oracle(
    const std::vector<float>& input, int position, float base) {
    const int head_dim = static_cast<int>(input.size());
    const int half = head_dim / 2;
    std::vector<float> output(head_dim);
    for (int pair = 0; pair < half; ++pair) {
        const float frequency = 1.0f / std::pow(base, static_cast<float>(2 * pair) / head_dim);
        const float angle = position * frequency;
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        const float first = input[pair];
        const float second = input[pair + half];
        output[pair] = first * cosine - second * sine;
        output[pair + half] = second * cosine + first * sine;
    }
    return output;
}

std::shared_ptr<Tensor> make_fp16_tensor(
    const std::shared_ptr<CUDAAllocator>& allocator,
    const std::vector<int>& shape,
    const std::vector<float>& values) {
    auto tensor = std::make_shared<Tensor>(allocator, shape, DataType::kDataTypeFP16);
    tensor->tensor_set_device_type(DeviceType::kDeviceCUDA);
    std::vector<__half> half_values(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        half_values[i] = __float2half(values[i]);
    }
    check_cuda(
        cudaMemcpy(
            tensor->raw_data_ptr(), half_values.data(),
            half_values.size() * sizeof(__half), cudaMemcpyHostToDevice),
        "Copy RoPE input to GPU");
    return tensor;
}

std::vector<float> copy_fp16(const std::shared_ptr<Tensor>& tensor) {
    std::vector<__half> half_values(tensor->tensor_total_elements());
    check_cuda(
        cudaMemcpy(
            half_values.data(), tensor->raw_data_ptr(),
            half_values.size() * sizeof(__half), cudaMemcpyDeviceToHost),
        "Copy RoPE output to host");
    std::vector<float> values(half_values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = __half2float(half_values[i]);
    }
    return values;
}

void require_near(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    const std::string& label) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(label + " size mismatch");
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > 2.0e-2f) {
            throw std::runtime_error(
                label + " mismatch at index " + std::to_string(i) +
                ": actual=" + std::to_string(actual[i]) +
                " expected=" + std::to_string(expected[i]));
        }
    }
}

}  // namespace

int main() {
    try {
        constexpr float base = 1000000.0f;
        ModelConfig config{};
        config.dim = 8;
        config.head = 1;
        config.kv_head = 1;
        auto allocator = std::make_shared<CUDAAllocator>();

        const std::vector<float> decode_input = {1, 2, 3, 4, 5, 6, 7, 8};
        auto decode_q = make_fp16_tensor(allocator, {1, 8}, decode_input);
        auto decode_k = make_fp16_tensor(allocator, {1, 8}, decode_input);
        rope_tensor(decode_q, decode_k, config, 7, base);
        check_cuda(cudaDeviceSynchronize(), "Synchronize decode RoPE");
        const std::vector<float> decode_expected = qwen_rope_oracle(decode_input, 7, base);
        require_near(copy_fp16(decode_q), decode_expected, "decode Qwen RoPE");
        require_near(copy_fp16(decode_k), decode_expected, "decode Qwen K RoPE");

        const std::vector<float> row_zero = {1, 2, 3, 4, 5, 6, 7, 8};
        const std::vector<float> row_one = {11, 12, 13, 14, 15, 16, 17, 18};
        std::vector<float> prefill_input = row_zero;
        prefill_input.insert(prefill_input.end(), row_one.begin(), row_one.end());
        auto prefill_q = make_fp16_tensor(allocator, {2, 8}, prefill_input);
        auto prefill_k = make_fp16_tensor(allocator, {2, 8}, prefill_input);
        rope_prefill_cuda(prefill_q, prefill_k, config, 2, 0, base);
        check_cuda(cudaDeviceSynchronize(), "Synchronize prefill RoPE");

        std::vector<float> prefill_expected = row_zero;
        const std::vector<float> row_one_expected = qwen_rope_oracle(row_one, 1, base);
        prefill_expected.insert(prefill_expected.end(), row_one_expected.begin(), row_one_expected.end());
        require_near(copy_fp16(prefill_q), prefill_expected, "prefill Qwen Q RoPE");
        require_near(copy_fp16(prefill_k), prefill_expected, "prefill Qwen K RoPE");

        std::cout << "PASS: decode and prefill RoPE match Qwen split-half layout\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
