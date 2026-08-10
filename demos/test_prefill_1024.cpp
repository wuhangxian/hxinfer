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
    auto cpu_alloc = std::make_shared<CPUAllocator>();
    auto cuda_alloc = std::make_shared<CUDAAllocator>();
    ModelConfig config;
    auto model = LlamaWeightLoader::load("/root/dorianwu/models/llama2-7b/llama2-7b-safetensors", config, cpu_alloc, cuda_alloc);
    Llama7BTokenizer tokenizer("/root/dorianwu/models/llama2-7b/llama2-7b-safetensors/tokenizer.model");
    // Read pre-tokenized ShareGPT (1024 token IDs)
    std::ifstream f("/tmp/sharegpt_token_ids.txt");
    std::stringstream ss; ss << f.rdbuf();
    std::string s = ss.str();
    std::vector<int> tokens;
    std::stringstream ss2(s);
    std::string id;
    while(std::getline(ss2, id, (char)-1)) {
        // split by comma
        size_t pos = 0;
        while(pos < id.size()) {
            size_t next = id.find((char)44, pos);
            if(next == std::string::npos) next = id.size();
            tokens.push_back(std::stoi(id.substr(pos, next-pos)));
            pos = next + 1;
        }
        break;
    }
    tokens.insert(tokens.begin(), tokenizer.bos_id());
    int S = std::min((int)tokens.size(), 1025);
    tokens.resize(S);
    std::cout << "Tokens: " << S << std::endl;
    // Batch prefill
    auto input_gpu = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{S}, DataType::kDataTypeFP32);
    auto logits = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{S, config.vocab_size}, DataType::kDataTypeFP16);
    input_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
    logits->tensor_set_device_type(DeviceType::kDeviceCUDA);
    auto input_cpu = std::make_shared<Tensor>(cpu_alloc, std::vector<int>{S}, DataType::kDataTypeFP32);
    int* p = input_cpu->tensor_data_ptr<int>();
    for(int i=0;i<S;i++) p[i]=tokens[i];
    cudaMemcpy(input_gpu->raw_data_ptr(), input_cpu->raw_data_ptr(), S*sizeof(int), cudaMemcpyHostToDevice);
    model->forward_prefill(input_gpu, logits, S);
    cudaDeviceSynchronize();
    // Get last token
    auto last_row = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP16);
    last_row->tensor_set_device_type(DeviceType::kDeviceCUDA);
    cudaMemcpy(last_row->raw_data_ptr(),
               static_cast<char*>(logits->raw_data_ptr()) + (S-1)*config.vocab_size*sizeof(__half),
               config.vocab_size*sizeof(__half), cudaMemcpyDeviceToDevice);
    int next = argmax_tensor(last_row);
    std::cout << "Batch prefill next token: " << next << " [" << tokenizer.decode(next) << "]" << std::endl;
    // Now decode 5 tokens to see if output is sane
    auto in1_cpu = std::make_shared<Tensor>(cpu_alloc, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto in1_gpu = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto logits_d = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP16);
    in1_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
    logits_d->tensor_set_device_type(DeviceType::kDeviceCUDA);
    int pos = S;
    std::cout << "Decode: ";
    for(int i=0;i<10;i++){
        std::cout << tokenizer.decode(next) << std::flush;
        int* pp = in1_cpu->tensor_data_ptr<int>();
        pp[0]=next;
        cudaMemcpy(in1_gpu->raw_data_ptr(), in1_cpu->raw_data_ptr(), sizeof(int), cudaMemcpyHostToDevice);
        model->forward(in1_gpu, logits_d, pos);
        next = argmax_tensor(logits_d);
        pos++;
    }
    std::cout << std::endl;
    return 0;
}
