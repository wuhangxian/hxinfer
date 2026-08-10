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

void check_tensor(const char* name, std::shared_ptr<Tensor>& t) {
    int n = std::min((int)t->tensor_total_elements(), 10);
    if (t->tensor_data_type() == DataType::kDataTypeFP16) {
        const __half* p = t->tensor_data_ptr<__half>();
        bool has_nan = false, has_inf = false, all_zero = true;
        for (int i = 0; i < n; i++) {
            float v = __half2float(p[i]);
            if (std::isnan(v)) has_nan = true;
            if (std::isinf(v)) has_inf = true;
            if (v != 0) all_zero = false;
        }
        std::cout << name << ": nan=" << has_nan << " inf=" << has_inf << " all_zero=" << all_zero
                  << " first=" << __half2float(p[0]) << " size=" << t->tensor_total_elements() << std::endl;
    }
}

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
    for(int S : {256, 512, 1024}){
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
        std::cout << "\n=== S=" << S << " ===" << std::endl;
        model->forward_prefill(in_gpu, logits, S);
        cudaError_t err = cudaDeviceSynchronize();
        if(err != cudaSuccess) {
            std::cout << "CUDA ERROR after prefill: " << cudaGetErrorString(err) << std::endl;
        }
        check_tensor("logits", logits);
        auto last_row = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP16);
        last_row->tensor_set_device_type(DeviceType::kDeviceCUDA);
        cudaMemcpy(last_row->raw_data_ptr(),
                   static_cast<char*>(logits->raw_data_ptr()) + (S-1)*config.vocab_size*sizeof(__half),
                   config.vocab_size*sizeof(__half), cudaMemcpyDeviceToDevice);
        int next = argmax_tensor(last_row);
        std::cout << "next_token=" << next << " [" << tokenizer.decode(next) << "]" << std::endl;
    }
    return 0;
}
