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
    // Read ShareGPT token IDs
    std::ifstream f("/tmp/sharegpt_token_ids.txt");
    std::stringstream ss; ss << f.rdbuf();
    std::string s = ss.str();
    std::vector<int> tokens;
    size_t pos = 0;
    while(pos < s.size()){
        size_t next = s.find(44, pos); // comma
        if(next == std::string::npos) next = s.size();
        tokens.push_back(std::stoi(s.substr(pos, next-pos)));
        pos = next + 1;
    }
    tokens.insert(tokens.begin(), tok.bos_id());
    int S = std::min((int)tokens.size(), 1025);
    tokens.resize(S);
    // Prefill
    auto in_gpu = std::make_shared<Tensor>(cu, std::vector<int>{S}, DataType::kDataTypeFP32);
    in_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
    auto in_cpu = std::make_shared<Tensor>(cpu, std::vector<int>{S}, DataType::kDataTypeFP32);
    int* p = in_cpu->tensor_data_ptr<int>();
    for(int i=0;i<S;i++) p[i]=tokens[i];
    cudaMemcpy(in_gpu->raw_data_ptr(), in_cpu->raw_data_ptr(), S*sizeof(int), cudaMemcpyHostToDevice);
    auto logits = std::make_shared<Tensor>(cu, std::vector<int>{S, config.vocab_size}, DataType::kDataTypeFP16);
    logits->tensor_set_device_type(DeviceType::kDeviceCUDA);
    model->forward_prefill(in_gpu, logits, S);
    cudaDeviceSynchronize();
    // Check for NaN
    std::vector<__half> h(S * config.vocab_size);
    cudaMemcpy(h.data(), logits->raw_data_ptr(), (size_t)S * config.vocab_size * sizeof(__half), cudaMemcpyDeviceToHost);
    int nan = 0;
    for(int i=0;i<(int)h.size();i++) if(std::isnan(__half2float(h[i]))) nan++;
    std::cout << "S=1025: nan=" << nan << "/" << h.size() << std::endl;
    // Decode 5 tokens
    auto in1_cpu = std::make_shared<Tensor>(cpu, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto in1_gpu = std::make_shared<Tensor>(cu, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto logits_d = std::make_shared<Tensor>(cu, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP16);
    in1_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
    logits_d->tensor_set_device_type(DeviceType::kDeviceCUDA);
    int next = 0;
    // Get last row
    auto last_row = std::make_shared<Tensor>(cu, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP16);
    last_row->tensor_set_device_type(DeviceType::kDeviceCUDA);
    cudaMemcpy(last_row->raw_data_ptr(),
               static_cast<char*>(logits->raw_data_ptr()) + (S-1)*config.vocab_size*sizeof(__half),
               config.vocab_size*sizeof(__half), cudaMemcpyDeviceToDevice);
    next = argmax_tensor(last_row);
    std::cout << "First decode token: " << next << " [" << tok.decode(next) << "]" << std::endl;
    int pos_d = S;
    for(int i=0;i<5;i++){
        std::cout << tok.decode(next);
        int* pp = in1_cpu->tensor_data_ptr<int>();
        pp[0]=next;
        cudaMemcpy(in1_gpu->raw_data_ptr(), in1_cpu->raw_data_ptr(), sizeof(int), cudaMemcpyHostToDevice);
        model->forward(in1_gpu, logits_d, pos_d);
        next = argmax_tensor(logits_d);
        pos_d++;
    }
    std::cout << std::endl;
    return 0;
}
