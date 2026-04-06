#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <string>
#include "cuda_runtime.h"

#include "base/allocator.h"
#include "base/config.h"
#include "tensor/tensor.h"
#include "op/math_ops.h"
#include "model/llama_model.h"
#include "loader/llama15m_loader.h"
#include "loader/llama_tokenizer.h"

using namespace hxinfer;

void run_llama_15m_gpu() {
    auto cpu_allocator = std::make_shared<CPUAllocator>();
    auto cuda_allocator = std::make_shared<CUDAAllocator>();

    std::string model_path = "models/stories15M.bin";
    std::string tokenizer_path = "models/tokenizer.bin";
    ModelConfig config;

    // 1. 加载模型到 GPU
    std::cout << "\n>>> [1/3] 正在加载 LLaMA 15M 权重到 GPU..." << std::endl;
    auto llama_engine = Llama15MLoader::load_model_cuda(model_path, config, cpu_allocator, cuda_allocator);

    // 2. 加载分词器
    std::cout << "\n>>> [2/3] 正在初始化分词器..." << std::endl;
    LlamaTokenizer tokenizer(tokenizer_path, config.vocab_size);

    // 3. 准备 I/O Tensor
    std::cout << "\n>>> [3/3] 分配 GPU 端 I/O 张量..." << std::endl;
    // input: 在 CPU 端写 token_id, 然后拷贝到 GPU
    auto input_cpu = std::make_shared<Tensor>(cpu_allocator, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto input_gpu = std::make_shared<Tensor>(cuda_allocator, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto logits_gpu = std::make_shared<Tensor>(cuda_allocator, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP32);

    std::cout << "\n---------------- GPU 童话故事生成开始 ----------------\n" << std::endl;

    int current_token_id = 1; // BOS
    int max_generate_step = 200;
    int actual_steps = 0;

    auto total_start = std::chrono::high_resolution_clock::now();

    for (int pos = 0; pos < max_generate_step; pos++) {
        auto step_start = std::chrono::high_resolution_clock::now();

        // 把 token_id 写到 CPU tensor, 再拷贝到 GPU
        int* ptr = input_cpu->tensor_data_ptr<int>();
        ptr[0] = current_token_id;
        cudaMemcpy(input_gpu->raw_data_ptr(), input_cpu->raw_data_ptr(),
                   sizeof(float), cudaMemcpyHostToDevice);

        // 前向传播 (全部在 GPU 上执行)
        llama_engine->forward(input_gpu, logits_gpu, pos);

        // argmax (在 GPU 上计算, 结果拷回 CPU)
        int next_token_id = argmax_tensor(logits_gpu);

        std::string word = tokenizer.decode(next_token_id);
        actual_steps = pos + 1;

        if (next_token_id == 2) {
            std::cout << "\n\n[GPU] 接收到 <EOS> 结束符，停止生成。" << std::endl;
            break;
        }

        auto step_end = std::chrono::high_resolution_clock::now();
        auto step_us = std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start).count();
        float tok_per_sec = 1000000.0f / (step_us == 0 ? 1 : step_us);

        if (word == "\n") {
            std::cout << word << std::flush;
        } else {
            std::cout << word << "\033[90m[" << tok_per_sec << " tok/s]\033[0m\n" << std::flush;
        }

        current_token_id = next_token_id;
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();

    std::cout << "\n\n---------------- GPU 故事生成结束 ----------------\n";
    std::cout << "Total time: " << ms / 1000.0f << " s\n";
    std::cout << "Steps: " << actual_steps << "\n";
    std::cout << "Avg speed: " << (float)actual_steps / (ms / 1000.0f) << " tokens/s\n";
}

void run_llama_15m_cpu() {
    auto cpu_allocator = std::make_shared<CPUAllocator>();

    std::string model_path = "models/stories15M.bin";
    std::string tokenizer_path = "models/tokenizer.bin";
    ModelConfig config;

    std::cout << "\n>>> [1/3] 正在加载 LLaMA 15M 权重到 CPU..." << std::endl;
    auto llama_engine = Llama15MLoader::load_model(model_path, config, cpu_allocator);

    std::cout << "\n>>> [2/3] 正在初始化分词器..." << std::endl;
    LlamaTokenizer tokenizer(tokenizer_path, config.vocab_size);

    std::cout << "\n>>> [3/3] 分配 CPU 端 I/O 张量..." << std::endl;
    auto input_tensor = std::make_shared<Tensor>(cpu_allocator, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto logits_tensor = std::make_shared<Tensor>(cpu_allocator, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP32);

    std::cout << "\n---------------- CPU 童话故事生成开始 ----------------\n" << std::endl;

    int current_token_id = 1;
    int max_generate_step = 200;
    int actual_steps = 0;

    auto total_start = std::chrono::high_resolution_clock::now();

    for (int pos = 0; pos < max_generate_step; pos++) {
        auto step_start = std::chrono::high_resolution_clock::now();

        int* input_ptr = input_tensor->tensor_data_ptr<int>();
        input_ptr[0] = current_token_id;

        llama_engine->forward(input_tensor, logits_tensor, pos);

        int next_token_id = argmax_tensor(logits_tensor);
        std::string word = tokenizer.decode(next_token_id);
        actual_steps = pos + 1;

        if (next_token_id == 2) {
            std::cout << "\n\n[CPU] 接收到 <EOS> 结束符，停止生成。" << std::endl;
            break;
        }

        auto step_end = std::chrono::high_resolution_clock::now();
        auto step_us = std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start).count();
        float tok_per_sec = 1000000.0f / (step_us == 0 ? 1 : step_us);

        if (word == "\n") {
            std::cout << word << std::flush;
        } else {
            std::cout << word << "\033[90m[" << tok_per_sec << " tok/s]\033[0m\n" << std::flush;
        }

        current_token_id = next_token_id;
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();

    std::cout << "\n\n---------------- CPU 故事生成结束 ----------------\n";
    std::cout << "Total time: " << ms / 1000.0f << " s\n";
    std::cout << "Steps: " << actual_steps << "\n";
    std::cout << "Avg speed: " << (float)actual_steps / (ms / 1000.0f) << " tokens/s\n";
}

int main(int argc, char* argv[]) {
    std::cout << "===================================================" << std::endl;
    std::cout << "HXInfer - CPU vs GPU 性能对比 Demo" << std::endl;
    std::cout << "===================================================" << std::endl;

    std::string mode = "both";
    if (argc > 1) {
        mode = argv[1]; // "cpu", "gpu", or "both"
    }

    try {
        if (mode == "cpu" || mode == "both") {
            std::cout << "\n==================== CPU 推理 ====================\n";
            run_llama_15m_cpu();
        }
        if (mode == "gpu" || mode == "both") {
            std::cout << "\n==================== GPU 推理 ====================\n";
            run_llama_15m_gpu();
        }
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
    }

    return 0;
}
