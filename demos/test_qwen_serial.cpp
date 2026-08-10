#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
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
#include "qwen_weight_loader.h"
#include "qwen_tokenizer.h"
using namespace hxinfer;
int main(){
    auto cpu = std::make_shared<CPUAllocator>();
    auto cu = std::make_shared<CUDAAllocator>();
    ModelConfig config;
    auto model = QwenWeightLoader::load("/root/dorianwu/models/Qwen2.5-7B-Instruct", config, cpu, cu);
    QwenTokenizer tok("/root/dorianwu/models/Qwen2.5-7B-Instruct/tokenizer.json");
    std::string prompt = "Once upon a time";
    std::vector<int> tokens = tok.encode(prompt);
    tokens.insert(tokens.begin(), tok.bos_id());
    int S = tokens.size();
    std::cout << "Prompt: " << prompt << " (" << S << " tokens)" << std::endl;
    // Serial prefill
    auto in1_cpu = std::make_shared<Tensor>(cpu, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto in1_gpu = std::make_shared<Tensor>(cu, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto logits1 = std::make_shared<Tensor>(cu, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP16);
    in1_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
    logits1->tensor_set_device_type(DeviceType::kDeviceCUDA);
    for(int pos=0;pos<S;pos++){
        int* p = in1_cpu->tensor_data_ptr<int>();
        p[0]=tokens[pos];
        cudaMemcpy(in1_gpu->raw_data_ptr(), in1_cpu->raw_data_ptr(), sizeof(int), cudaMemcpyHostToDevice);
        model->forward(in1_gpu, logits1, pos);
    }
    int serial_next = argmax_tensor(logits1);
    std::cout << "Serial next: " << serial_next << " [" << tok.decode(serial_next) << "]" << std::endl;
    // Decode 20 tokens
    std::cout << "Serial output: ";
    int next = serial_next;
    int pos = S;
    for(int i=0;i<20;i++){
        std::cout << tok.decode(next) << std::flush;
        int* p = in1_cpu->tensor_data_ptr<int>();
        p[0]=next;
        cudaMemcpy(in1_gpu->raw_data_ptr(), in1_cpu->raw_data_ptr(), sizeof(int), cudaMemcpyHostToDevice);
        model->forward(in1_gpu, logits1, pos);
        next = argmax_tensor(logits1);
        pos++;
    }
    std::cout << std::endl;
    // Batch prefill
    auto in_gpu = std::make_shared<Tensor>(cu, std::vector<int>{S}, DataType::kDataTypeFP32);
    auto logits = std::make_shared<Tensor>(cu, std::vector<int>{S, config.vocab_size}, DataType::kDataTypeFP16);
    in_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
    logits->tensor_set_device_type(DeviceType::kDeviceCUDA);
    auto in_cpu = std::make_shared<Tensor>(cpu, std::vector<int>{S}, DataType::kDataTypeFP32);
    int* pp = in_cpu->tensor_data_ptr<int>();
    for(int i=0;i<S;i++) pp[i]=tokens[i];
    cudaMemcpy(in_gpu->raw_data_ptr(), in_cpu->raw_data_ptr(), S*sizeof(int), cudaMemcpyHostToDevice);
    model->forward_prefill(in_gpu, logits, S);
    cudaDeviceSynchronize();
    auto last_row = std::make_shared<Tensor>(cu, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP16);
    last_row->tensor_set_device_type(DeviceType::kDeviceCUDA);
    cudaMemcpy(last_row->raw_data_ptr(),
               static_cast<char*>(logits->raw_data_ptr()) + (S-1)*config.vocab_size*sizeof(__half),
               config.vocab_size*sizeof(__half), cudaMemcpyDeviceToDevice);
    int batch_next = argmax_tensor(last_row);
    std::cout << "Batch next: " << batch_next << " [" << tok.decode(batch_next) << "]" << std::endl;
    if(serial_next == batch_next) std::cout << "MATCH" << std::endl;
    else std::cout << "MISMATCH" << std::endl;
    return 0;
}
