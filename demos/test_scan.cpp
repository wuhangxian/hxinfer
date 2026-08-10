#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <cstdlib>
#include <fstream>
#include <sstream>
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
    std::ifstream f("/root/dorianwu/sharegpt_tokens.txt");
    std::stringstream ss; ss << f.rdbuf();
    std::string s = ss.str();
    std::vector<int> tokens;
    size_t pos=0;
    while(pos<s.size()){size_t n=s.find(44,pos); if(n==std::string::npos)n=s.size(); tokens.push_back(std::stoi(s.substr(pos,n-pos))); pos=n+1;}
    tokens.insert(tokens.begin(), tok.bos_id());
    for(int S = 300; S <= 400; S += 10){
        if((int)tokens.size() < S) break;
        std::vector<int> t(tokens.begin(), tokens.begin()+S);
        auto in_gpu = std::make_shared<Tensor>(cu, std::vector<int>{S}, DataType::kDataTypeFP32);
        auto logits = std::make_shared<Tensor>(cu, std::vector<int>{S, config.vocab_size}, DataType::kDataTypeFP16);
        in_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
        logits->tensor_set_device_type(DeviceType::kDeviceCUDA);
        auto in_cpu = std::make_shared<Tensor>(cpu, std::vector<int>{S}, DataType::kDataTypeFP32);
        int* pp = in_cpu->tensor_data_ptr<int>();
        for(int i=0;i<S;i++) pp[i]=t[i];
        cudaMemcpy(in_gpu->raw_data_ptr(), in_cpu->raw_data_ptr(), S*sizeof(int), cudaMemcpyHostToDevice);
        model->forward_prefill(in_gpu, logits, S);
        cudaDeviceSynchronize();
        auto last_row = std::make_shared<Tensor>(cu, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP16);
        last_row->tensor_set_device_type(DeviceType::kDeviceCUDA);
        cudaMemcpy(last_row->raw_data_ptr(),
                   static_cast<char*>(logits->raw_data_ptr()) + (S-1)*config.vocab_size*sizeof(__half),
                   config.vocab_size*sizeof(__half), cudaMemcpyDeviceToDevice);
        int next = argmax_tensor(last_row);
        std::vector<__half> h(config.vocab_size);
        cudaMemcpy(h.data(), last_row->raw_data_ptr(), config.vocab_size*sizeof(__half), cudaMemcpyDeviceToHost);
        int nan=0;
        for(int i=0;i<config.vocab_size;i++) if(std::isnan(__half2float(h[i]))) nan++;
        std::cout << "S=" << S << " next=" << next << " nan=" << nan << " [" << tok.decode(next) << "]";
        if(nan>0) std::cout << " <=== NaN";
        std::cout << std::endl;
    }
    return 0;
}
