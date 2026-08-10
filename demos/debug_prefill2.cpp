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
    std::string prompt;
    for(int i=0;i<100;i++) prompt += "The quick brown fox jumps over the lazy dog. ";
    std::vector<int> all_tokens = tokenizer.encode(prompt);
    all_tokens.insert(all_tokens.begin(), tokenizer.bos_id());
    int S = 256;
    std::vector<int> tokens(all_tokens.begin(), all_tokens.begin() + S);
    auto in_gpu = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{S}, DataType::kDataTypeFP32);
    auto logits = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{S, config.vocab_size}, DataType::kDataTypeFP16);
    in_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
    logits->tensor_set_device_type(DeviceType::kDeviceCUDA);
    auto in_cpu = std::make_shared<Tensor>(cpu_alloc, std::vector<int>{S}, DataType::kDataTypeFP32);
    int* p = in_cpu->tensor_data_ptr<int>();
    for(int i=0;i<S;i++) p[i]=tokens[i];
    cudaMemcpy(in_gpu->raw_data_ptr(), in_cpu->raw_data_ptr(), S*sizeof(int), cudaMemcpyHostToDevice);
    std::cout << "Running forward_prefill S=" << S << std::endl;
    model->forward_prefill(in_gpu, logits, S);
    cudaError_t err = cudaDeviceSynchronize();
    std::cout << "Sync result: " << cudaGetErrorString(err) << std::endl;
    // Check logits
    __half* logits_ptr = logits->tensor_data_ptr<__half>();
    int total = S * config.vocab_size;
    int nan_count = 0, zero_count = 0;
    // Check last row only
    int last_row_start = (S-1) * config.vocab_size;
    float max_val = -1e30f;
    int max_idx = 0;
    for(int i=0;i<config.vocab_size;i++){
        float v = __half2float(logits_ptr[last_row_start + i]);
        if(std::isnan(v)) nan_count++;
        if(v == 0) zero_count++;
        if(v > max_val){ max_val = v; max_idx = i; }
    }
    std::cout << "Last row: nan_count=" << nan_count << " zero_count=" << zero_count
              << " max_val=" << max_val << " max_idx=" << max_idx
              << " [" << tokenizer.decode(max_idx) << "]" << std::endl;
    return 0;
}
