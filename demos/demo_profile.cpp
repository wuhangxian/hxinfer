#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <string>
#include "cuda_runtime.h"
#include "base/allocator.h"
#include "base/config.h"
#include "tensor/tensor.h"
#include "op/math_ops.h"
#include "llama_weight_loader.h"
#include "loader/llama7b_loader.h"

using namespace hxinfer;

int main(){
    const std::string DATA_DIR = "/workspace/whx/hxinfer-data";

    auto cpu_alloc  = std::make_shared<CPUAllocator>();
    auto cuda_alloc = std::make_shared<CUDAAllocator>();

    std::cout << ">>> 加载 LLaMA-2 7B 权重...\n";
    ModelConfig config;
    auto model = Llama7BLoader::load(DATA_DIR, config, cpu_alloc, cuda_alloc);

    auto input_cpu = std::make_shared<Tensor>(cpu_alloc, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto input_gpu = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto logits    = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP32);
    input_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
    logits->tensor_set_device_type(DeviceType::kDeviceCUDA);

    // Warmup
    int* ptr = input_cpu->tensor_data_ptr<int>();
    ptr[0] = 1;
    cudaMemcpy(input_gpu->raw_data_ptr(), input_cpu->raw_data_ptr(), sizeof(int), cudaMemcpyHostToDevice);
    for(int i=0; i<5; i++){
        model->forward(input_gpu, logits, i);
        int tok = argmax_tensor(logits);
        (void)tok;
    }
    cudaDeviceSynchronize();

    const int NUM_STEPS = 100;

    // ── Test 1: 纯模型 forward（无采样） ──
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1); cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    for(int i=0; i<NUM_STEPS; i++){
        model->forward(input_gpu, logits, 5 + i);
    }
    cudaEventRecord(stop1); cudaEventSynchronize(stop1);
    float ms1; cudaEventElapsedTime(&ms1, start1, stop1);

    // ── Test 2: forward + GPU sample_tensor ──
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2); cudaEventCreate(&stop2);
    cudaEventRecord(start2);
    for(int i=0; i<NUM_STEPS; i++){
        model->forward(input_gpu, logits, 5 + i);
        int tok = sample_tensor(logits, 0.8f, 0.9f);
        (void)tok;
    }
    cudaEventRecord(stop2); cudaEventSynchronize(stop2);
    float ms2; cudaEventElapsedTime(&ms2, start2, stop2);

    // ── Test 3: forward + argmax（无采样，对比） ──
    cudaEvent_t start3, stop3;
    cudaEventCreate(&start3); cudaEventCreate(&stop3);
    cudaEventRecord(start3);
    for(int i=0; i<NUM_STEPS; i++){
        model->forward(input_gpu, logits, 5 + i);
        int tok = argmax_tensor(logits);
        (void)tok;
    }
    cudaEventRecord(stop3); cudaEventSynchronize(stop3);
    float ms3; cudaEventElapsedTime(&ms3, start3, stop3);

    std::cout << "\n========== Decode 性能对比 (" << NUM_STEPS << " 步) ==========\n";
    std::cout << "  纯 forward:            " << NUM_STEPS/(ms1/1000.0f) << " tok/s (" << ms1/NUM_STEPS << " ms/step)\n";
    std::cout << "  forward + argmax:      " << NUM_STEPS/(ms3/1000.0f) << " tok/s (" << ms3/NUM_STEPS << " ms/step)\n";
    std::cout << "  forward + GPU sample:  " << NUM_STEPS/(ms2/1000.0f) << " tok/s (" << ms2/NUM_STEPS << " ms/step)\n";
    std::cout << "  PyTorch 参考值:        ~48.3 tok/s\n";
    std::cout << "\n  采样开销: " << (ms2-ms1)/NUM_STEPS << " ms/step (vs forward only)\n";

    return 0;
}
