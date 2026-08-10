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
#include "qwen_weight_loader.h"
#include "qwen_tokenizer.h"

using namespace hxinfer;

int main(int argc, char** argv){
    const char* env_data = std::getenv("HXINFER_DATA_DIR");
    const std::string DATA_DIR = env_data ? std::string(env_data) : "/workspace/models/Qwen2.5-7B";
    const std::string TOKEN_PATH = DATA_DIR + "/tokenizer.json";
    const std::string PROMPT = "Once upon a time";
    const int INPUT_LEN = 1024;
    const int OUTPUT_LEN = 1024;

    auto cpu_alloc = std::make_shared<CPUAllocator>();
    auto cuda_alloc = std::make_shared<CUDAAllocator>();

    std::cout << "\n>>> Loading Qwen2.5-7B..." << std::endl;
    ModelConfig config;
    auto model = QwenWeightLoader::load(DATA_DIR, config, cpu_alloc, cuda_alloc);

    std::cout << ">>> Loading tokenizer..." << std::endl;
    QwenTokenizer tokenizer(TOKEN_PATH);

    std::string base = "Hello world ";
    std::vector<int> tokens;
    while((int)tokens.size() < INPUT_LEN) {
        auto t = tokenizer.encode(base);
        tokens.insert(tokens.end(), t.begin(), t.end());
    }
    tokens.insert(tokens.begin(), tokenizer.bos_id());
    tokens.resize(INPUT_LEN);

    std::cout << "\nInput: " << INPUT_LEN << " tokens" << std::endl;
    std::cout << "Output: " << OUTPUT_LEN << " tokens" << std::endl;

    auto input_ids_cpu = std::make_shared<Tensor>(cpu_alloc, std::vector<int>{INPUT_LEN}, DataType::kDataTypeFP32);
    auto input_ids_gpu = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{INPUT_LEN}, DataType::kDataTypeFP32);
    input_ids_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
    int* ids_ptr = input_ids_cpu->tensor_data_ptr<int>();
    for(int i = 0; i < INPUT_LEN; i++) ids_ptr[i] = tokens[i];
    cudaMemcpy(input_ids_gpu->raw_data_ptr(), input_ids_cpu->raw_data_ptr(),
               INPUT_LEN * sizeof(int), cudaMemcpyHostToDevice);

    auto logits_prefill = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{INPUT_LEN, config.vocab_size}, DataType::kDataTypeFP16);
    logits_prefill->tensor_set_device_type(DeviceType::kDeviceCUDA);

    auto input_single_cpu = std::make_shared<Tensor>(cpu_alloc, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto input_single_gpu = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto logits_decode = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP16);
    input_single_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
    logits_decode->tensor_set_device_type(DeviceType::kDeviceCUDA);

    auto t0 = std::chrono::high_resolution_clock::now();
    model->forward_prefill(input_ids_gpu, logits_prefill, INPUT_LEN);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto last_row = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP16);
    last_row->tensor_set_device_type(DeviceType::kDeviceCUDA);
    cudaMemcpy(last_row->raw_data_ptr(),
               static_cast<char*>(logits_prefill->raw_data_ptr()) + (INPUT_LEN - 1) * config.vocab_size * sizeof(__half),
               config.vocab_size * sizeof(__half), cudaMemcpyDeviceToDevice);
    int next_token = argmax_tensor(last_row);

    int generated = 0;
    int pos = INPUT_LEN;
    std::string output_text;

    for(int w = 0; w < 5; w++){
        int* ptr = input_single_cpu->tensor_data_ptr<int>();
        ptr[0] = next_token;
        cudaMemcpy(input_single_gpu->raw_data_ptr(), input_single_cpu->raw_data_ptr(),
                   sizeof(int), cudaMemcpyHostToDevice);
        model->forward(input_single_gpu, logits_decode, pos);
        next_token = argmax_tensor(logits_decode);
        pos++;
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    while(generated < OUTPUT_LEN){
        output_text += tokenizer.decode(next_token);
        int* ptr = input_single_cpu->tensor_data_ptr<int>();
        ptr[0] = next_token;
        cudaMemcpy(input_single_gpu->raw_data_ptr(), input_single_cpu->raw_data_ptr(),
                   sizeof(int), cudaMemcpyHostToDevice);
        model->forward(input_single_gpu, logits_decode, pos);
        next_token = argmax_tensor(logits_decode);
        pos++;
        generated++;
    }
    cudaDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();

    double decode_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double total_ms = prefill_ms + decode_ms;

    std::cout << "\n--- Generated Output (first 500 chars) ---" << std::endl;
    std::cout << output_text.substr(0, 500) << std::endl;

    std::cout << "\n============ Result ============" << std::endl;
    std::cout << "Prefill: " << INPUT_LEN << " tokens / " << prefill_ms/1000.0 << " s = " << (INPUT_LEN / (prefill_ms/1000.0)) << " tok/s" << std::endl;
    std::cout << "  TTFT:   " << prefill_ms << " ms" << std::endl;
    std::cout << "Decode:  " << OUTPUT_LEN << " tokens / " << decode_ms/1000.0 << " s = " << (OUTPUT_LEN / (decode_ms/1000.0)) << " tok/s" << std::endl;
    std::cout << "  ITL:    " << (decode_ms / OUTPUT_LEN) << " ms" << std::endl;
    std::cout << "Total:   " << (INPUT_LEN + OUTPUT_LEN) << " tokens / " << total_ms/1000.0 << " s" << std::endl;
    std::cout << "Throughput: " << ((INPUT_LEN + OUTPUT_LEN) / (total_ms/1000.0)) << " tok/s" << std::endl;
    std::cout << "===============================" << std::endl;
    return 0;
}
