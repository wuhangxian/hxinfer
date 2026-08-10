#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <string>
#include <cstdlib>
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
    const char* env_data = std::getenv("HXINFER_DATA_DIR");
    const std::string DATA_DIR = env_data ? std::string(env_data) : "/workspace/models/llama2-7b";
    const std::string TOKEN_PATH = DATA_DIR + "/tokenizer.model";
    const std::string PROMPT = "Once upon a time";
    const int MAX_NEW_TOKENS = 100;

    auto cpu_alloc  = std::make_shared<CPUAllocator>();
    auto cuda_alloc = std::make_shared<CUDAAllocator>();

    std::cout << "\n>>> Loading model..." << std::endl;
    ModelConfig config;
    auto model = LlamaWeightLoader::load(DATA_DIR, config, cpu_alloc, cuda_alloc);

    std::cout << ">>> Loading tokenizer..." << std::endl;
    Llama7BTokenizer tokenizer(TOKEN_PATH);

    // Encode prompt
    std::vector<int> tokens = tokenizer.encode(PROMPT);
    tokens.insert(tokens.begin(), tokenizer.bos_id());
    int prompt_len = tokens.size();

    std::cout << "\nPrompt: \"" << PROMPT << "\" (" << prompt_len << " tokens)" << std::endl;

    // Batch prefill
    auto input_ids_cpu = std::make_shared<Tensor>(cpu_alloc, std::vector<int>{prompt_len}, DataType::kDataTypeFP32);
    auto input_ids_gpu = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{prompt_len}, DataType::kDataTypeFP32);
    input_ids_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);

    int* ids_ptr = input_ids_cpu->tensor_data_ptr<int>();
    for(int i = 0; i < prompt_len; i++) ids_ptr[i] = tokens[i];
    cudaMemcpy(input_ids_gpu->raw_data_ptr(), input_ids_cpu->raw_data_ptr(),
               prompt_len * sizeof(int), cudaMemcpyHostToDevice);

    auto logits_prefill = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{prompt_len, config.vocab_size}, DataType::kDataTypeFP16);
    logits_prefill->tensor_set_device_type(DeviceType::kDeviceCUDA);

    std::cout << "\n--- Batch Prefill ---" << std::endl;
    model->forward_prefill(input_ids_gpu, logits_prefill, prompt_len);
    cudaDeviceSynchronize();

    // Get last token
    auto last_row = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP16);
    last_row->tensor_set_device_type(DeviceType::kDeviceCUDA);
    cudaMemcpy(last_row->raw_data_ptr(),
               static_cast<char*>(logits_prefill->raw_data_ptr()) + (prompt_len - 1) * config.vocab_size * sizeof(__half),
               config.vocab_size * sizeof(__half), cudaMemcpyDeviceToDevice);
    int next_token = argmax_tensor(last_row);

    // Decode
    auto input_single_cpu = std::make_shared<Tensor>(cpu_alloc, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto input_single_gpu = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto logits_decode = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP16);
    input_single_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
    logits_decode->tensor_set_device_type(DeviceType::kDeviceCUDA);

    std::cout << "\n--- Generation ---\n" << PROMPT << std::flush;

    int generated = 0;
    int pos = prompt_len;

    while(generated < MAX_NEW_TOKENS){
        std::string word = tokenizer.decode(next_token);
        std::cout << word << std::flush;
        if(next_token == tokenizer.eos_id()) break;

        int* ptr = input_single_cpu->tensor_data_ptr<int>();
        ptr[0] = next_token;
        cudaMemcpy(input_single_gpu->raw_data_ptr(), input_single_cpu->raw_data_ptr(),
                   sizeof(int), cudaMemcpyHostToDevice);
        model->forward(input_single_gpu, logits_decode, pos);
        next_token = argmax_tensor(logits_decode);
        pos++;
        generated++;
    }

    std::cout << "\n\n--- Done ---\n" << std::endl;
    return 0;
}
