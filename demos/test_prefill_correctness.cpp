#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <cstdlib>
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "base/allocator.h"
#include "base/config.h"
#include "tensor/tensor.h"
#include "op/math_ops.h"
#include "model/causal_lm_model.h"
#include "llama_weight_loader.h"
#include "llama7b_tokenizer.h"
using namespace hxinfer;
int main(){
    auto cpu_alloc = std::make_shared<CPUAllocator>();
    auto cuda_alloc = std::make_shared<CUDAAllocator>();
    ModelConfig config;
    auto model = LlamaWeightLoader::load("/root/dorianwu/models/llama2-7b/llama2-7b-safetensors", config, cpu_alloc, cuda_alloc);
    Llama7BTokenizer tokenizer("/root/dorianwu/models/llama2-7b/llama2-7b-safetensors/tokenizer.model");
    std::vector<int> tokens = tokenizer.encode("Once upon a time");
    tokens.insert(tokens.begin(), tokenizer.bos_id());
    int S = tokens.size();
    std::cout << "Prompt tokens: " << S << std::endl;
    // Method 1: serial prefill (old way)
    auto input1_cpu = std::make_shared<Tensor>(cpu_alloc, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto input1_gpu = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto logits1 = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP16);
    input1_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
    logits1->tensor_set_device_type(DeviceType::kDeviceCUDA);
    for(int pos = 0; pos < S; pos++){
        int* p = input1_cpu->tensor_data_ptr<int>();
        p[0] = tokens[pos];
        cudaMemcpy(input1_gpu->raw_data_ptr(), input1_cpu->raw_data_ptr(), sizeof(int), cudaMemcpyHostToDevice);
        model->forward(input1_gpu, logits1, pos);
    }
    int serial_next = argmax_tensor(logits1);
    std::cout << "Serial prefill next token: " << serial_next << " [" << tokenizer.decode(serial_next) << "]" << std::endl;
    // Method 2: batch prefill (new way)
    auto input2_cpu = std::make_shared<Tensor>(cpu_alloc, std::vector<int>{S}, DataType::kDataTypeFP32);
    auto input2_gpu = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{S}, DataType::kDataTypeFP32);
    auto logits2 = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{S, config.vocab_size}, DataType::kDataTypeFP16);
    input2_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
    logits2->tensor_set_device_type(DeviceType::kDeviceCUDA);
    int* p2 = input2_cpu->tensor_data_ptr<int>();
    for(int i = 0; i < S; i++) p2[i] = tokens[i];
    cudaMemcpy(input2_gpu->raw_data_ptr(), input2_cpu->raw_data_ptr(), S * sizeof(int), cudaMemcpyHostToDevice);
    model->forward_prefill(input2_gpu, logits2, S);
    cudaDeviceSynchronize();
    auto last_row = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP16);
    last_row->tensor_set_device_type(DeviceType::kDeviceCUDA);
    cudaMemcpy(last_row->raw_data_ptr(),
               static_cast<char*>(logits2->raw_data_ptr()) + (S-1) * config.vocab_size * sizeof(__half),
               config.vocab_size * sizeof(__half), cudaMemcpyDeviceToDevice);
    int batch_next = argmax_tensor(last_row);
    std::cout << "Batch prefill next token: " << batch_next << " [" << tokenizer.decode(batch_next) << "]" << std::endl;
    if(serial_next == batch_next)
        std::cout << "MATCH! Prefill is correct." << std::endl;
    else
        std::cout << "MISMATCH! Prefill has a bug." << std::endl;
    return 0;
}
