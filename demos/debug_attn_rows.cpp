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
    int S = 512;
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
    int vocab = config.vocab_size;
    std::vector<__half> h_logits(S * vocab);
    cudaMemcpy(h_logits.data(), logits->raw_data_ptr(), (size_t)S * vocab * sizeof(__half), cudaMemcpyDeviceToHost);
    // Check which rows have NaN
    int first_nan_row = -1;
    int last_nan_row = -1;
    int nan_rows = 0;
    for(int row=0;row<S;row++){
        bool has_nan = false;
        for(int i=0;i<vocab;i++){
            if(std::isnan(__half2float(h_logits[row*vocab+i]))){ has_nan = true; break; }
        }
        if(has_nan){
            if(first_nan_row < 0) first_nan_row = row;
            last_nan_row = row;
            nan_rows++;
        }
    }
    std::cout << "S=512: nan_rows=" << nan_rows << "/" << S
              << " first=" << first_nan_row << " last=" << last_nan_row << std::endl;
    return 0;
}
