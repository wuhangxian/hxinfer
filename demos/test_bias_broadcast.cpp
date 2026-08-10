#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hxinfer {
extern "C" void add_bias_broadcast_fp16(__half* data, const __half* bias, int total, int width);
extern "C" void add_bias_broadcast_fp32(float* data, const float* bias, int total, int width);
}  // namespace hxinfer

namespace {

constexpr int kGuardElements = 37;

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

bool same_half_bits(__half lhs, __half rhs) {
    return std::memcmp(&lhs, &rhs, sizeof(__half)) == 0;
}

void run_fp16_case(int rows, int width) {
    const int total = rows * width;
    const __half guard = __float2half(123.0f);
    std::vector<__half> host_data(total + 2 * kGuardElements, guard);
    std::vector<__half> host_bias(width);

    for (int i = 0; i < total; ++i) {
        host_data[kGuardElements + i] = __float2half(static_cast<float>((i % 17) - 8) * 0.25f);
    }
    for (int i = 0; i < width; ++i) {
        host_bias[i] = __float2half(static_cast<float>((i % 13) - 6) * 0.125f);
    }

    __half* device_data = nullptr;
    __half* device_bias = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_data), host_data.size() * sizeof(__half)), "cudaMalloc(data fp16)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_bias), host_bias.size() * sizeof(__half)), "cudaMalloc(bias fp16)");
    check_cuda(cudaMemcpy(device_data, host_data.data(), host_data.size() * sizeof(__half), cudaMemcpyHostToDevice), "cudaMemcpy(data fp16 H2D)");
    check_cuda(cudaMemcpy(device_bias, host_bias.data(), host_bias.size() * sizeof(__half), cudaMemcpyHostToDevice), "cudaMemcpy(bias fp16 H2D)");

    hxinfer::add_bias_broadcast_fp16(device_data + kGuardElements, device_bias, total, width);
    check_cuda(cudaGetLastError(), "add_bias_broadcast_fp16 launch");
    check_cuda(cudaDeviceSynchronize(), "add_bias_broadcast_fp16 synchronize");

    std::vector<__half> actual(host_data.size());
    check_cuda(cudaMemcpy(actual.data(), device_data, actual.size() * sizeof(__half), cudaMemcpyDeviceToHost), "cudaMemcpy(data fp16 D2H)");

    for (int i = 0; i < kGuardElements; ++i) {
        if (!same_half_bits(actual[i], guard) || !same_half_bits(actual[kGuardElements + total + i], guard)) {
            throw std::runtime_error("FP16 guard modified for rows=" + std::to_string(rows) + " width=" + std::to_string(width));
        }
    }
    for (int i = 0; i < total; ++i) {
        const float expected = __half2float(host_data[kGuardElements + i]) + __half2float(host_bias[i % width]);
        const float observed = __half2float(actual[kGuardElements + i]);
        if (std::fabs(observed - expected) > 1.0e-3f) {
            throw std::runtime_error("FP16 oracle mismatch at element " + std::to_string(i));
        }
    }

    check_cuda(cudaFree(device_bias), "cudaFree(bias fp16)");
    check_cuda(cudaFree(device_data), "cudaFree(data fp16)");
}

void run_fp32_case(int rows, int width) {
    const int total = rows * width;
    constexpr float guard = 123456.25f;
    std::vector<float> host_data(total + 2 * kGuardElements, guard);
    std::vector<float> host_bias(width);

    for (int i = 0; i < total; ++i) {
        host_data[kGuardElements + i] = static_cast<float>((i % 17) - 8) * 0.25f;
    }
    for (int i = 0; i < width; ++i) {
        host_bias[i] = static_cast<float>((i % 13) - 6) * 0.125f;
    }

    float* device_data = nullptr;
    float* device_bias = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_data), host_data.size() * sizeof(float)), "cudaMalloc(data fp32)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_bias), host_bias.size() * sizeof(float)), "cudaMalloc(bias fp32)");
    check_cuda(cudaMemcpy(device_data, host_data.data(), host_data.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy(data fp32 H2D)");
    check_cuda(cudaMemcpy(device_bias, host_bias.data(), host_bias.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy(bias fp32 H2D)");

    hxinfer::add_bias_broadcast_fp32(device_data + kGuardElements, device_bias, total, width);
    check_cuda(cudaGetLastError(), "add_bias_broadcast_fp32 launch");
    check_cuda(cudaDeviceSynchronize(), "add_bias_broadcast_fp32 synchronize");

    std::vector<float> actual(host_data.size());
    check_cuda(cudaMemcpy(actual.data(), device_data, actual.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy(data fp32 D2H)");

    for (int i = 0; i < kGuardElements; ++i) {
        if (actual[i] != guard || actual[kGuardElements + total + i] != guard) {
            throw std::runtime_error("FP32 guard modified for rows=" + std::to_string(rows) + " width=" + std::to_string(width));
        }
    }
    for (int i = 0; i < total; ++i) {
        const float expected = host_data[kGuardElements + i] + host_bias[i % width];
        const float observed = actual[kGuardElements + i];
        if (std::fabs(observed - expected) > 1.0e-6f) {
            throw std::runtime_error("FP32 oracle mismatch at element " + std::to_string(i));
        }
    }

    check_cuda(cudaFree(device_bias), "cudaFree(bias fp32)");
    check_cuda(cudaFree(device_data), "cudaFree(data fp32)");
}

}  // namespace

int main() {
    try {
        for (int rows : {1, 2}) {
            for (int width : {3584, 512}) {
                run_fp16_case(rows, width);
                run_fp32_case(rows, width);
                std::cout << "PASS rows=" << rows << " width=" << width << " fp16+fp32 oracle+guards\n";
            }
        }
        std::cout << "All bias broadcast regression cases passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
