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
#include "loader/llama7b_loader.h"
#include "loader/llama7b_tokenizer.h"

using namespace hxinfer;

int main(int argc, char** argv){
    const std::string DATA_DIR   = "/workspace/whx/hxinfer-data";
    const std::string TOKEN_PATH = DATA_DIR + "/Yarn-Llama-2-7b-128k/tokenizer.model";
    const std::string PROMPT     = "Once upon a time";
    const int MAX_NEW_TOKENS     = 200;

    // 采样参数（可通过命令行覆盖）
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

    // ── 1. 加载模型 ───────────────────────────────────────────────────────
    std::cout << "\n>>> [1/3] 加载 LLaMA-2 7B 权重...\n";
    ModelConfig config;
    auto model = Llama7BLoader::load(DATA_DIR, config, cpu_alloc, cuda_alloc);

    // ── 2. 加载分词器 ─────────────────────────────────────────────────────
    std::cout << "\n>>> [2/3] 加载 Tokenizer...\n";
    Llama7BTokenizer tokenizer(TOKEN_PATH);

    // ── 3. 准备 I/O Tensor ────────────────────────────────────────────────
    std::cout << "\n>>> [3/3] 准备 I/O Tensor...\n";
    auto input_cpu = std::make_shared<Tensor>(cpu_alloc, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto input_gpu = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto logits    = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP32);
    input_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
    logits->tensor_set_device_type(DeviceType::kDeviceCUDA);

    // ── 4. 编码 prompt ────────────────────────────────────────────────────
    std::vector<int> tokens = tokenizer.encode(PROMPT);
    tokens.insert(tokens.begin(), tokenizer.bos_id());   // 手动加 BOS

    std::cout << "\n提示词: \"" << PROMPT << "\"\n";
    std::cout << "Token 数量: " << tokens.size() << "\n";
    std::cout << "采样参数: temperature=" << temperature << ", top_p=" << top_p << "\n";
    std::cout << "\n---------------- 生成开始 ----------------\n";
    std::cout << PROMPT << std::flush;

    auto t_start = std::chrono::high_resolution_clock::now();

    // ── 5. Prefill：先把 prompt tokens 过一遍模型 ─────────────────────────
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

    // ── 6. Decode：自回归生成 ─────────────────────────────────────────────
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

    std::cout << "\n\n---------------- 生成结束 ----------------\n";
    std::cout << "生成 token 数: " << generated << "\n";
    std::cout << "总耗时: " << ms/1000.0f << " s\n";
    std::cout << "平均速度: " << generated / (ms/1000.0f) << " tok/s\n";

    return 0;
}
