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
#include "model/llama_model.h"
#include "loader/llama7b_loader.h"
#include "loader/llama7b_tokenizer.h"
using namespace hxinfer;
int main(){
    auto cpu = std::make_shared<CPUAllocator>();
    auto cu = std::make_shared<CUDAAllocator>();
    ModelConfig config;
    auto model = Llama7BLoader::load("/root/dorianwu/models/llama2-7b/llama2-7b-safetensors", config, cpu, cu);
    Llama7BTokenizer tok("/root/dorianwu/models/llama2-7b/llama2-7b-safetensors/tokenizer.model");
    std::string prompt;
    for(int i=0;i<100;i++) prompt += "The quick brown fox jumps over the lazy dog. ";
    std::vector<int> all_tokens = tok.encode(prompt);
    all_tokens.insert(all_tokens.begin(), tok.bos_id());
    auto in1_cpu = std::make_shared<Tensor>(cpu, std::vector<int>{1}, DataType::kDataTypeFP32);
       auto in1_gpu = std::make_shared<Tensor>(cu, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto logits1 = std::make_shared<Tensor>(cu, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP16);
    in1_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
    logits1->tensor_set_device_type(DeviceType::kDeviceCUDA);
    for(int S : {256, 512}){
        if((int)all_tokens.size() < S) break;
        std::vector<int> tokens(all_tokens.begin(), all_tokens.begin() + S);
        for(int pos=0; pos<S; pos++){
            int* p = in1_cpu->tensor_data_ptr<int>();
            p[0] = tokens[pos];
            cudaMemcpy(in1_gpu->raw_data_ptr(), in1_cpu->raw_data_ptr(), sizeof(int), cudaMemcpyHostToDevice);
            model->forward(in1_gpu, logits1, pos);
        }
        int next = argmax_tensor(logits1);
        std::cout << "Main Serial S=" << S << " next_token=" << next << " [" << tok.decode(next) << "]" << std::endl;
    }
    return 0;
}
