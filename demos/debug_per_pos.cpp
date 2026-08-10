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
#include "layer/transformer.h"
#include "model/causal_lm_model.h"
#include "llama_weight_loader.h"
#include "llama7b_tokenizer.h"
using namespace hxinfer;

int check_nan(std::shared_ptr<Tensor>& t, int n=100) {
    if(t->tensor_data_type() != DataType::kDataTypeFP16) return -1;
    int total = std::min((int)t->tensor_total_elements(), n);
    std::vector<__half> h(total);
    cudaMemcpy(h.data(), t->raw_data_ptr(), total*sizeof(__half), cudaMemcpyDeviceToHost);
    int nan = 0;
    for(int i=0;i<total;i++) if(std::isnan(__half2float(h[i]))) nan++;
    return nan;
}

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
    // Check positions 200, 250, 300, 350, 400, 410, 420, 430, 435, 438, 439, 440
    int cps[] = {200, 250, 300, 350, 380, 390, 395, 398, 399, 400, 401, 402, 410, 420, 430, 439, 440, 500, 511};
    for(int ci=0; ci<19; ci++){
        int target = cps[ci];
        if(target >= (int)all_tokens.size()) break;
        for(int pos=0; pos<=target; pos++){
            int* p = in_cpu->tensor_data_ptr<int>();
            p[0] = all_tokens[pos];
            cudaMemcpy(in_gpu->raw_data_ptr(), in_cpu->raw_data_ptr(), sizeof(int), cudaMemcpyHostToDevice);
            model->forward(in_gpu, logits, pos);
        }
        int nan = check_nan(logits, config.vocab_size);
        int next = (nan > 0) ? -1 : argmax_tensor(logits);
        std::cout << "pos=" << target << " nan=" << nan << " next=" << next;
        if(nan > 0) std::cout << " <=== NaN";
        std::cout << std::endl;
    }
    return 0;
}
