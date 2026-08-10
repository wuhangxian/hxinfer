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
#include "layer/embedding.h"
#include "layer/rmsnorm.h"
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
    int S = 16;
    tokens.resize(S);
    // Embedding
    auto ids_gpu = std::make_shared<Tensor>(cu, std::vector<int>{S}, DataType::kDataTypeFP32);
    ids_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
    auto ids_cpu = std::make_shared<Tensor>(cpu, std::vector<int>{S}, DataType::kDataTypeFP32);
    int* p = ids_cpu->tensor_data_ptr<int>();
    for(int i=0;i<S;i++) p[i]=tokens[i];
    cudaMemcpy(ids_gpu->raw_data_ptr(), ids_cpu->raw_data_ptr(), S*sizeof(int), cudaMemcpyHostToDevice);
    auto embed = std::make_shared<Tensor>(cu, std::vector<int>{S, config.dim}, DataType::kDataTypeFP16);
    embed->tensor_set_device_type(DeviceType::kDeviceCUDA);
    embedding_tensor(ids_gpu, model->get_blocks()[0]->get_attention()->get_k_cache(), embed);
    // Wait, need actual embedding weight. Let me just do the full prefill and check layer 0.
    // Actually, just run the full model prefill and serial prefill, compare logits at each layer.
    // Simplest: run forward_prefill with S=16 and forward() 16 times, compare final logits.
    auto logits_batch = std::make_shared<Tensor>(cu, std::vector<int>{S, config.vocab_size}, DataType::kDataTypeFP16);
    logits_batch->tensor_set_device_type(DeviceType::kDeviceCUDA);
    model->forward_prefill(ids_gpu, logits_batch, S);
    cudaDeviceSynchronize();
    // Serial
    auto in1_cpu = std::make_shared<Tensor>(cpu, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto in1_gpu = std::make_shared<Tensor>(cu, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto logits_serial = std::make_shared<Tensor>(cu, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP16);
    in1_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
    logits_serial->tensor_set_device_type(DeviceType::kDeviceCUDA);
    for(int pos=0;pos<S;pos++){
        int* pp = in1_cpu->tensor_data_ptr<int>();
        pp[0]=tokens[pos];
        cudaMemcpy(in1_gpu->raw_data_ptr(), in1_cpu->raw_data_ptr(), sizeof(int), cudaMemcpyHostToDevice);
        model->forward(in1_gpu, logits_serial, pos);
    }
    // Compare last token logits
    int batch_next = argmax_tensor(logits_batch);
    int serial_next = argmax_tensor(logits_serial);
    std::cout << "S=16 batch=" << batch_next << " serial=" << serial_next;
    if(batch_next == serial_next) std::cout << " MATCH" << std::endl;
    else std::cout << " MISMATCH" << std::endl;
    return 0;
}
