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
    // Test with increasing sequence lengths
    std::string prompt = "The quick brown fox jumps over the lazy dog. ";
    std::vector<int> all_tokens = tokenizer.encode(prompt);
    all_tokens.insert(all_tokens.begin(), tokenizer.bos_id());
    for(int S : {5, 16, 32, 64, 128, 256, 512, 1024}){
        if((int)all_tokens.size() < S) break;
        std::vector<int> tokens(all_tokens.begin(), all_tokens.begin() + S);
        // Serial prefill
        auto in1_cpu = std::make_shared<Tensor>(cpu_alloc, std::vector<int>{1}, DataType::kDataTypeFP32);
        auto in1_gpu = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1}, DataType::kDataTypeFP32);
        auto logits1 = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP16);
        in1_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
        logits1->tensor_set_device_type(DeviceType::kDeviceCUDA);
        for(int pos=0; pos<S; pos++){
            int* p = in1_cpu->tensor_data_ptr<int>();
            p[0] = tokens[pos];
            cudaMemcpy(in1_gpu->raw_data_ptr(), in1_cpu->raw_data_ptr(), sizeof(int), cudaMemcpyHostToDevice);
            model->forward(in1_gpu, logits1, pos);
        }
        int serial_next = argmax_tensor(logits1);
        // Batch prefill
        auto in2_gpu = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{S}, DataType::kDataTypeFP32);
        auto logits2 = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{S, config.vocab_size}, DataType::kDataTypeFP16);
        in2_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
        logits2->tensor_set_device_type(DeviceType::kDeviceCUDA);
        auto in2_cpu = std::make_shared<Tensor>(cpu_alloc, std::vector<int>{S}, DataType::kDataTypeFP32);
        int* p2 = in2_cpu->tensor_data_ptr<int>();
        for(int i=0;i<S;i++) p2[i]=tokens[i];
        cudaMemcpy(in2_gpu->raw_data_ptr(), in2_cpu->raw_data_ptr(), S*sizeof(int), cudaMemcpyHostToDevice);
        model->forward_prefill(in2_gpu, logits2, S);
        cudaDeviceSynchronize();
        auto last_row = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP16);
        last_row->tensor_set_device_type(DeviceType::kDeviceCUDA);
        cudaMemcpy(last_row->raw_data_ptr(),
                   static_cast<char*>(logits2->raw_data_ptr()) + (S-1)*config.vocab_size*sizeof(__half),
                   config.vocab_size*sizeof(__half), cudaMemcpyDeviceToDevice);
        int batch_next = argmax_tensor(last_row);
        bool match = (serial_next == batch_next);
        std::cout << "S=" << S << " serial=" << serial_next << " batch=" << batch_next
                  << " " << (match ? "MATCH" : "MISMATCH") << std::endl;
    }
    return 0;
}
