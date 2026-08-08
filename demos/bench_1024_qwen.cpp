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
#include "qwen_weight_loader.h"
#include "llama_weight_loader.h"
#include "qwen_tokenizer.h"

using namespace hxinfer;

int main(int argc, char** argv){
    const char* env_data = std::getenv("HXINFER_DATA_DIR");
    std::string DATA_DIR = env_data ? std::string(env_data) : "/workspace/models/Qwen2.5-7B";
    std::string TOKEN_PATH = DATA_DIR + "/tokenizer.json";

    int INPUT_LEN  = 1024;
    int OUTPUT_LEN = 1024;
    if(argc > 1) INPUT_LEN  = std::atoi(argv[1]);
    if(argc > 2) OUTPUT_LEN = std::atoi(argv[2]);

    auto cpu_alloc  = std::make_shared<CPUAllocator>();
    auto cuda_alloc = std::make_shared<CUDAAllocator>();

    std::cout << "\n>>> Loading model from safetensors..." << std::endl;
    ModelConfig config;
    auto model = QwenWeightLoader::load(DATA_DIR, config, cpu_alloc, cuda_alloc);

    std::cout << ">>> Loading tokenizer..." << std::endl;
    QwenTokenizer tokenizer(TOKEN_PATH);

    auto input_cpu = std::make_shared<Tensor>(cpu_alloc, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto input_gpu = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto logits    = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP16);
    input_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
    logits->tensor_set_device_type(DeviceType::kDeviceCUDA);

    std::string base = "Hello world ";
    std::vector<int> tokens;
    while((int)tokens.size() < INPUT_LEN + 1) {
        auto t = tokenizer.encode(base);
        tokens.insert(tokens.end(), t.begin(), t.end());
    }
    tokens.insert(tokens.begin(), tokenizer.bos_id());
    tokens.resize(INPUT_LEN + 1);

    std::cout << "\nInput: " << INPUT_LEN << " tokens" << std::endl;
    std::cout << "Output: " << OUTPUT_LEN << " tokens" << std::endl;
    std::cout << "\n--- Benchmark ---" << std::endl;

    auto t_prefill_start = std::chrono::high_resolution_clock::now();
    int next_token = tokens[0];
    for(int pos = 0; pos < (int)tokens.size(); pos++){
        int* ptr = input_cpu->tensor_data_ptr<int>();
        ptr[0] = tokens[pos];
        cudaMemcpy(input_gpu->raw_data_ptr(), input_cpu->raw_data_ptr(),
                   sizeof(int), cudaMemcpyHostToDevice);
        model->forward(input_gpu, logits, pos);
        if(pos == (int)tokens.size() - 1)
            next_token = argmax_tensor(logits);
    }
    cudaDeviceSynchronize();
    auto t_prefill_end = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();

    int generated = 0;
    int pos = (int)tokens.size();
    for(int w = 0; w < 5 && generated < OUTPUT_LEN; w++){
        int* ptr = input_cpu->tensor_data_ptr<int>();
        ptr[0] = next_token;
        cudaMemcpy(input_gpu->raw_data_ptr(), input_cpu->raw_data_ptr(),
                   sizeof(int), cudaMemcpyHostToDevice);
        model->forward(input_gpu, logits, pos);
        next_token = argmax_tensor(logits);
        pos++;
    }

    auto t_decode_start = std::chrono::high_resolution_clock::now();
    while(generated < OUTPUT_LEN){
        int* ptr = input_cpu->tensor_data_ptr<int>();
        ptr[0] = next_token;
        cudaMemcpy(input_gpu->raw_data_ptr(), input_cpu->raw_data_ptr(),
                   sizeof(int), cudaMemcpyHostToDevice);
        model->forward(input_gpu, logits, pos);
        next_token = argmax_tensor(logits);
        pos++;
        generated++;
    }
    cudaDeviceSynchronize();
    auto t_decode_end = std::chrono::high_resolution_clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();

    double prefill_s = prefill_ms / 1000.0;
    double decode_s  = decode_ms / 1000.0;
    double total_s   = prefill_s + decode_s;

    std::cout << "\n============ Result ============" << std::endl;
    std::cout << "Prefill: " << INPUT_LEN << " tokens / " << prefill_s << " s = " << (INPUT_LEN / prefill_s) << " tok/s" << std::endl;
    std::cout << "  TTFT:   " << prefill_ms << " ms" << std::endl;
    std::cout << "Decode:  " << OUTPUT_LEN << " tokens / " << decode_s << " s = " << (OUTPUT_LEN / decode_s) << " tok/s" << std::endl;
    std::cout << "  ITL:    " << (decode_ms / OUTPUT_LEN) << " ms" << std::endl;
    std::cout << "Total:   " << (INPUT_LEN + OUTPUT_LEN) << " tokens / " << total_s << " s" << std::endl;
    std::cout << "Throughput: " << ((INPUT_LEN + OUTPUT_LEN) / total_s) << " tok/s" << std::endl;
    std::cout << "===============================" << std::endl;
    return 0;
}
