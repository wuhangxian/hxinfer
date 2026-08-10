#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <cstdlib>
#include <cmath>
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
    auto cpu = std::make_shared<CPUAllocator>();
    auto cu = std::make_shared<CUDAAllocator>();
    ModelConfig config;
    auto model = LlamaWeightLoader::load("/root/dorianwu/models/llama2-7b/llama2-7b-safetensors", config, cpu, cu);
    Llama7BTokenizer tok("/root/dorianwu/models/llama2-7b/llama2-7b-safetensors/tokenizer.model");
    std::string prompt;
    for(int i=0;i<100;i++) prompt += "The quick brown fox jumps over the lazy dog. ";
    std::vector<int> all_tokens = tok.encode(prompt);
    all_tokens.insert(all_tokens.begin(), tok.bos_id());
    auto in_cpu = std::make_shared<Tensor>(cpu, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto in_gpu = std::make_shared<Tensor>(cu, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto logits = std::make_shared<Tensor>(cu, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP16);
    in_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
    logits->tensor_set_device_type(DeviceType::kDeviceCUDA);
    for(int pos=0; pos<512; pos++){
        int* p = in_cpu->tensor_data_ptr<int>();
        p[0] = all_tokens[pos];
        cudaMemcpy(in_gpu->raw_data_ptr(), in_cpu->raw_data_ptr(), sizeof(int), cudaMemcpyHostToDevice);
        model->forward(in_gpu, logits, pos);
        if(pos >= 250 && (pos <= 260 || pos % 50 == 0 || (pos >= 390 && pos <= 410))) {
            std::vector<__half> h(config.vocab_size);
            cudaMemcpy(h.data(), logits->raw_data_ptr(), config.vocab_size*sizeof(__half), cudaMemcpyDeviceToHost);
            int nan=0;
            for(int i=0;i<config.vocab_size;i++) if(std::isnan(__half2float(h[i]))) nan++;
            int next = (nan>0) ? -1 : argmax_tensor(logits);
            std::cout << "pos=" << pos << " nan=" << nan << " next=" << next;
            if(nan>0) std::cout << " <=== NaN";
            std::cout << std::endl;
        }
    }
    return 0;
}
