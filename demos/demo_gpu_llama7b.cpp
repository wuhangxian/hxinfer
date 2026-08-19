#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <string>
#include <cstdlib>
#include "cuda_runtime.h"
#include "base/allocator.h"
#include "base/config.h"
#include "tensor/tensor.h"
#include "op/math_ops.h"
#include "llama_weight_loader.h"
#include "llama_weight_loader.h"
#include "llama7b_tokenizer.h"

using namespace hxinfer;

int main(int argc, char** argv){
    const char* env_data = std::getenv("HXINFER_DATA_DIR");
    const std::string DATA_DIR   = env_data ? std::string(env_data) : "/workspace/models/llama2-7b";
    const std::string TOKEN_PATH = DATA_DIR + "/tokenizer.model";
    const std::string PROMPT     = "Once upon a time";
    const int MAX_NEW_TOKENS     = 200;

    float temperature = 0.8f;
    float top_p       = 0.9f;
    bool greedy       = false;
    if(argc > 1){
        std::string arg1 = argv[1];
        if(arg1 == "--greedy" || arg1 == "-g"){
            greedy = true;
        } else {
            temperature = std::stof(argv[1]);
        }
    }
    if(argc > 2) top_p = std::stof(argv[2]);

    auto cpu_alloc  = std::make_shared<CPUAllocator>();
    auto cuda_alloc = std::make_shared<CUDAAllocator>();

    std::cout << "\n>>> [1/3] Loading model from safetensors...\n";
    ModelConfig config;
    auto model = LlamaWeightLoader::load(DATA_DIR, config, cpu_alloc, cuda_alloc);

    std::cout << "\n>>> [2/3] Loading tokenizer...\n";
    Llama7BTokenizer tokenizer(TOKEN_PATH);

    std::cout << "\n>>> [3/3] Preparing I/O tensors...\n";
    auto input_cpu = std::make_shared<Tensor>(cpu_alloc, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto input_gpu = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto logits    = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1, config.vocab_size}, config.logits_dtype);
    input_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
    logits->tensor_set_device_type(DeviceType::kDeviceCUDA);

    std::vector<int> tokens = tokenizer.encode(PROMPT);
    tokens.insert(tokens.begin(), tokenizer.bos_id());

    std::cout << "\nPrompt: \"" << PROMPT << "\"\n";
    std::cout << "Tokens: " << tokens.size() << "\n";
    std::cout << "Sampling: temperature=" << temperature << ", top_p=" << top_p << "\n";
    std::cout << "\n--- Generation ---\n";
    std::cout << PROMPT << std::flush;

    auto t_start = std::chrono::high_resolution_clock::now();

    int next_token = tokens[0];
    for(int pos=0; pos < (int)tokens.size(); pos++){
        int* ptr = input_cpu->tensor_data_ptr<int>();
        ptr[0]   = tokens[pos];
        cudaMemcpy(input_gpu->raw_data_ptr(), input_cpu->raw_data_ptr(),
                   sizeof(int), cudaMemcpyHostToDevice);
        model->forward(input_gpu, logits, pos);
        if(pos == (int)tokens.size()-1)
            next_token = sample_tensor(logits, temperature, top_p);
    }

    int generated = 0;
    int pos = (int)tokens.size();

    while(generated < MAX_NEW_TOKENS){
        std::string word = tokenizer.decode(next_token);
        std::cout << word << std::flush;
        if(next_token == tokenizer.eos_id()) break;

        int* ptr = input_cpu->tensor_data_ptr<int>();
        ptr[0]   = next_token;
        cudaMemcpy(input_gpu->raw_data_ptr(), input_cpu->raw_data_ptr(),
                   sizeof(int), cudaMemcpyHostToDevice);
        model->forward(input_gpu, logits, pos);
        next_token = sample_tensor(logits, temperature, top_p);
        pos++;
        generated++;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

    std::cout << "\n\n--- Done ---\n";
    std::cout << "Tokens generated: " << generated << "\n";
    std::cout << "Time: " << ms/1000.0f << " s\n";
    std::cout << "Speed: " << generated / (ms/1000.0f) << " tok/s\n";

    return 0;
}
