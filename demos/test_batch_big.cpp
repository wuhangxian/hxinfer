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
    // Build a long prompt with enough tokens
    std::string prompt;
    for(int i=0;i<100;i++) prompt += "The quick brown fox jumps over the lazy dog. ";
    std::vector<int> all_tokens = tokenizer.encode(prompt);
    all_tokens.insert(all_tokens.begin(), tokenizer.bos_id());
    std::cout << "Total tokens: " << all_tokens.size() << std::endl;
    for(int S : {16, 32, 64, 128, 256, 512, 1024}){
        if((int)all_tokens.size() < S) break;
        std::vector<int> tokens(all_tokens.begin(), all_tokens.begin() + S);
        auto in_gpu = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{S}, DataType::kDataTypeFP32);
        auto logits = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{S, config.vocab_size}, DataType::kDataTypeFP16);
        in_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
        logits->tensor_set_device_type(DeviceType::kDeviceCUDA);
        auto in_cpu = std::make_shared<Tensor>(cpu_alloc, std::vector<int>{S}, DataType::kDataTypeFP32);
        int* p = in_cpu->tensor_data_ptr<int>();
        for(int i=0;i<S;i++) p[i]=tokens[i];
        cudaMemcpy(in_gpu->raw_data_ptr(), in_cpu->raw_data_ptr(), S*sizeof(int), cudaMemcpyHostToDevice);
        model->forward_prefill(in_gpu, logits, S);
        cudaDeviceSynchronize();
        auto last_row = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP16);
        last_row->tensor_set_device_type(DeviceType::kDeviceCUDA);
        cudaMemcpy(last_row->raw_data_ptr(),
                   static_cast<char*>(logits->raw_data_ptr()) + (S-1)*config.vocab_size*sizeof(__half),
                   config.vocab_size*sizeof(__half), cudaMemcpyDeviceToDevice);
        int next = argmax_tensor(last_row);
        std::cout << "S=" << S << " next_token=" << next << " [" << tokenizer.decode(next) << "]" << std::endl;
    }
    return 0;
}
