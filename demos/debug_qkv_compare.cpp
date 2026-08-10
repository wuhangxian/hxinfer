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
#include "layer/attention.h"
#include "layer/transformer.h"
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
    std::string prompt = "The quick brown fox jumps over the lazy dog. ";
    std::vector<int> tokens = tok.encode(prompt);
    tokens.insert(tokens.begin(), tok.bos_id());
    int S = 8;
    tokens.resize(S);
    int dim = config.dim;
    // Serial: get embedding for each token, then QKV
    auto in1_cpu = std::make_shared<Tensor>(cpu, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto in1_gpu = std::make_shared<Tensor>(cu, std::vector<int>{1}, DataType::kDataTypeFP32);
    in1_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
    // Collect serial embeddings
    std::vector<__half> serial_embed(S * dim);
    for(int pos=0;pos<S;pos++){
        int* p = in1_cpu->tensor_data_ptr<int>();
        p[0]=tokens[pos];
        cudaMemcpy(in1_gpu->raw_data_ptr(), in1_cpu->raw_data_ptr(), sizeof(int), cudaMemcpyHostToDevice);
        // We need the embedding output, but model->forward does everything.
        // Instead, just call the embedding layer.
        // Actually, let me just compare the final logits.
    }
    // Run serial prefill
    auto logits1 = std::make_shared<Tensor>(cu, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP16);
    logits1->tensor_set_device_type(DeviceType::kDeviceCUDA);
    for(int pos=0;pos<S;pos++){
        int* p = in1_cpu->tensor_data_ptr<int>();
        p[0]=tokens[pos];
        cudaMemcpy(in1_gpu->raw_data_ptr(), in1_cpu->raw_data_ptr(), sizeof(int), cudaMemcpyHostToDevice);
        model->forward(in1_gpu, logits1, pos);
    }
    int serial_next = argmax_tensor(logits1);
    // Run batch prefill
    auto ids_gpu = std::make_shared<Tensor>(cu, std::vector<int>{S}, DataType::kDataTypeFP32);
    ids_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
    auto ids_cpu = std::make_shared<Tensor>(cpu, std::vector<int>{S}, DataType::kDataTypeFP32);
    int* pp = ids_cpu->tensor_data_ptr<int>();
    for(int i=0;i<S;i++) pp[i]=tokens[i];
    cudaMemcpy(ids_gpu->raw_data_ptr(), ids_cpu->raw_data_ptr(), S*sizeof(int), cudaMemcpyHostToDevice);
    auto logits2 = std::make_shared<Tensor>(cu, std::vector<int>{S, config.vocab_size}, DataType::kDataTypeFP16);
    logits2->tensor_set_device_type(DeviceType::kDeviceCUDA);
    model->forward_prefill(ids_gpu, logits2, S);
    cudaDeviceSynchronize();
    // Get last row
    auto last_row = std::make_shared<Tensor>(cu, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP16);
    last_row->tensor_set_device_type(DeviceType::kDeviceCUDA);
    cudaMemcpy(last_row->raw_data_ptr(),
               static_cast<char*>(logits2->raw_data_ptr()) + (S-1)*config.vocab_size*sizeof(__half),
               config.vocab_size*sizeof(__half), cudaMemcpyDeviceToDevice);
    int batch_next = argmax_tensor(last_row);
    std::cout << "S=8 serial=" << serial_next << " batch=" << batch_next;
    if(serial_next==batch_next) std::cout << " MATCH" << std::endl;
    else std::cout << " MISMATCH" << std::endl;
    return 0;
}
