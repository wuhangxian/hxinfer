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

int main(int argc, char** argv){
    const char* env_data = std::getenv("HXINFER_DATA_DIR");
    const std::string DATA_DIR = env_data ? std::string(env_data) : "/workspace/models/llama2-7b";
    const std::string TOKEN_PATH = DATA_DIR + "/tokenizer.model";

    int INPUT_LEN = 1024;
    int OUTPUT_LEN = 1024;
    if(argc > 1) INPUT_LEN = std::atoi(argv[1]);
    if(argc > 2) OUTPUT_LEN = std::atoi(argv[2]);

    auto cpu_alloc = std::make_shared<CPUAllocator>();
    auto cuda_alloc = std::make_shared<CUDAAllocator>();

    std::cout << "\n>>> Loading model from safetensors..." << std::endl;
    ModelConfig config;
    auto model = LlamaWeightLoader::load(DATA_DIR, config, cpu_alloc, cuda_alloc);

    std::cout << ">>> Loading tokenizer..." << std::endl;
    Llama7BTokenizer tokenizer(TOKEN_PATH);

    // Prepare input tensor for batch prefill: [S] int32 on GPU
    auto input_ids_cpu = std::make_shared<Tensor>(cpu_alloc, std::vector<int>{INPUT_LEN}, DataType::kDataTypeFP32);
    auto input_ids_gpu = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{INPUT_LEN}, DataType::kDataTypeFP32);
    input_ids_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);

    // Decode tensor: [1, vocab] for single token decode
    auto input_single_cpu = std::make_shared<Tensor>(cpu_alloc, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto input_single_gpu = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto logits_prefill = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{INPUT_LEN, config.vocab_size}, DataType::kDataTypeFP16);
    auto logits_decode = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP16);
    input_single_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
    logits_prefill->tensor_set_device_type(DeviceType::kDeviceCUDA);
    logits_decode->tensor_set_device_type(DeviceType::kDeviceCUDA);

    // Build prompt tokens
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
    std::cout << "\n--- Benchmark (Batch Prefill) ---" << std::endl;

    // ============ Batch Prefill ============
    int* ids_ptr = input_ids_cpu->tensor_data_ptr<int>();
    for(int i = 0; i < INPUT_LEN; i++) ids_ptr[i] = tokens[i];
    cudaMemcpy(input_ids_gpu->raw_data_ptr(), input_ids_cpu->raw_data_ptr(),
               INPUT_LEN * sizeof(int), cudaMemcpyHostToDevice);

    auto t_prefill_start = std::chrono::high_resolution_clock::now();
    model->forward_prefill(input_ids_gpu, logits_prefill, INPUT_LEN);
    cudaDeviceSynchronize();
    auto t_prefill_end = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();

    // Get last token from prefill output
    int next_token = argmax_tensor(logits_prefill);
    // argmax_tensor finds max in the whole tensor, but we need last row
    // For now, use the decode path for first token after prefill

    // Actually we need the logits of the last token from prefill output
    // logits_prefill is [INPUT_LEN, vocab_size], we need row INPUT_LEN-1
    // Let's do a simple cudaMemcpy of last row to a smaller tensor and argmax
    auto last_row = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP16);
    last_row->tensor_set_device_type(DeviceType::kDeviceCUDA);
    cudaMemcpy(last_row->raw_data_ptr(),
               static_cast<char*>(logits_prefill->raw_data_ptr()) + (INPUT_LEN - 1) * config.vocab_size * sizeof(__half),
               config.vocab_size * sizeof(__half), cudaMemcpyDeviceToDevice);
    next_token = argmax_tensor(last_row);

    // ============ Decode (with warmup) ============
    int generated = 0;
    int pos = INPUT_LEN;

    // Warmup 5 tokens
    for(int w = 0; w < 5 && generated < OUTPUT_LEN; w++){
        int* ptr = input_single_cpu->tensor_data_ptr<int>();
        ptr[0] = next_token;
        cudaMemcpy(input_single_gpu->raw_data_ptr(), input_single_cpu->raw_data_ptr(),
                   sizeof(int), cudaMemcpyHostToDevice);
        model->forward(input_single_gpu, logits_decode, pos);
        next_token = argmax_tensor(logits_decode);
        pos++;
    }

    auto t_decode_start = std::chrono::high_resolution_clock::now();
    while(generated < OUTPUT_LEN){
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
    auto t_decode_end = std::chrono::high_resolution_clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();

    double prefill_s = prefill_ms / 1000.0;
    double decode_s = decode_ms / 1000.0;
    double total_s = prefill_s + decode_s;

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
