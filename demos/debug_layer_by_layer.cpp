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
#include "layer/transformer.h"
#include "model/causal_lm_model.h"
#include "llama_weight_loader.h"
#include "llama7b_tokenizer.h"
using namespace hxinfer;
void check(const char* name, std::shared_ptr<Tensor>& t) {
    if(t->tensor_data_type() != DataType::kDataTypeFP16) { std::cout << name << " (not fp16)" << std::endl; return; }
    int n = t->tensor_total_elements();
    std::vector<__half> h(n);
    cudaMemcpy(h.data(), t->raw_data_ptr(), n*sizeof(__half), cudaMemcpyDeviceToHost);
    int nan=0; for(int i=0;i<n;i++) if(std::isnan(__half2float(h[i]))) nan++;
    std::cout << name << ": nan=" << nan << "/" << n;
    if(nan==0 && n>0) std::cout << " first=" << __half2float(h[0]);
    std::cout << std::endl;
}
int main(){
    auto cpu = std::make_shared<CPUAllocator>();
    auto cu = std::make_shared<CUDAAllocator>();
    ModelConfig config;
    auto model = LlamaWeightLoader::load("/root/dorianwu/models/llama2-7b/llama2-7b-safetensors", config, cpu, cu);
    Llama7BTokenizer tok("/root/dorianwu/models/llama2-7b/llama2-7b-safetensors/tokenizer.model");
    std::string p;
    for(int i=0;i<100;i++) p += "The quick brown fox jumps over the lazy dog. ";
    auto tokens = tok.encode(p);
    tokens.insert(tokens.begin(), tok.bos_id());
    int S = 512;
    tokens.resize(S);
    auto in_gpu = std::make_shared<Tensor>(cu, std::vector<int>{S}, DataType::kDataTypeFP32);
    in_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
    auto in_cpu = std::make_shared<Tensor>(cpu, std::vector<int>{S}, DataType::kDataTypeFP32);
    int* pp = in_cpu->tensor_data_ptr<int>();
    for(int i=0;i<S;i++) pp[i]=tokens[i];
    cudaMemcpy(in_gpu->raw_data_ptr(), in_cpu->raw_data_ptr(), S*sizeof(int), cudaMemcpyHostToDevice);
    auto logits = std::make_shared<Tensor>(cu, std::vector<int>{S, config.vocab_size}, DataType::kDataTypeFP16);
    logits->tensor_set_device_type(DeviceType::kDeviceCUDA);
    std::cout << "Running forward_prefill S=512..." << std::endl;
    model->forward_prefill(in_gpu, logits, S);
    cudaDeviceSynchronize();
    check("final_logits", logits);
    return 0;
}
