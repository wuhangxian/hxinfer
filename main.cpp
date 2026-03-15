#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <string>

// 1. 引入底层基建
#include "base/allocator.h"
#include "base/config.h"
#include "tensor/tensor.h"
#include "op/math_ops.h"

// 2. 引入你的三大核心架构模块！
#include "model/llama_model.h"
#include "loader/llama15m_loader.h"
#include "loader/llama_tokenizer.h"

using namespace hxinfer;

// =========================================================================
// 🌟 独立出来的运行函数：LLaMA 15M 专属点火舱 (以后定死不动了！)
// =========================================================================
void run_llama_15m_stories() {
    // 1. 准备物理内存分配器
    std::shared_ptr<Allocator> cpu_allocator = std::make_shared<CPUAllocator>();

    // 注意：请确保你的可执行文件同级目录（或指定的相对路径）下有 models 文件夹
    std::string model_path = "models/stories15M.bin";
    std::string tokenizer_path = "models/tokenizer.bin";
    ModelConfig config;

    // 2. 召唤装配厂：解析二进制，零拷贝拼装大模型！
    std::cout << "\n>>> [1/3] 正在加载 LLaMA 15M 权重与拓扑结构..." << std::endl;
    auto llama_engine = Llama15MLoader::load_model(model_path, config, cpu_allocator);

    // 3. 召唤翻译官：加载词表
    std::cout << "\n>>> [2/3] 正在初始化 LLaMA 分词器..." << std::endl;
    LlamaTokenizer tokenizer(tokenizer_path, config.vocab_size);

    // 4. 准备输入输出的 Tensor 容器
    std::cout << "\n>>> [3/3] 分配 I/O 张量，准备进入自回归生成..." << std::endl;
    auto input_tensor = std::make_shared<Tensor>(cpu_allocator, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto logits_tensor = std::make_shared<Tensor>(cpu_allocator, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP32);

    // 5. 正式开始推理！
    std::cout << "\n---------------- 📖 童话故事生成开始 ----------------\n" << std::endl;

    int current_token_id = 1; // 1 是 LLaMA 的 <BOS> (Begin of Sequence)
    int max_generate_step = 200; // 生成 200 个词

    auto total_start_time = std::chrono::high_resolution_clock::now();

    for (int pos = 0; pos < max_generate_step; pos++) {
        auto step_start = std::chrono::high_resolution_clock::now();

        // 将当前的 token 塞进 input_tensor
    // 将当前的 token 塞进 input_tensor（🌟 修正：按底层要求的 int 格式强行写入！）
        int* input_ptr = input_tensor->tensor_data_ptr<int>();
        input_ptr[0] = current_token_id; // 绝对不要转成 float，原汁原味的 int 喂进去！

        // ⚡ 核心前向传播！数据瞬间穿透 6 层 Transformer Block！
        llama_engine->forward(input_tensor, logits_tensor, pos);

        // 🔍 从 32000 个概率打分里，找出得分最高的词的 ID (贪心解码)
        int next_token_id = 0;
        // 如果你的 math_ops.h 里是封装在 MathOps 类里的，记得加上 MathOps::
        next_token_id=argmax_tensor(logits_tensor);

        // 🗣️ 解码并打印
        std::string word = tokenizer.decode(next_token_id);

        // 🛑 核心刹车逻辑：如果模型吐出了 <EOS> 结束符，立刻终止生成！
        if (next_token_id == 2) {
            std::cout << "\n\n[系统提示] 接收到 <EOS> 结束符，模型自然停止生成。" << std::endl;
            break;
        }
        if (word == "\n") {
            std::cout << word << std::flush;
        } else {
            // 计算瞬间速度
            auto step_end = std::chrono::high_resolution_clock::now();
            auto step_us = std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start).count();
            float tok_per_sec = 1000000.0f / (step_us == 0 ? 1 : step_us);

            // 极客风输出：带上灰色速度指标
            std::cout << word << "\033[90m[" << tok_per_sec << " token/s]"<<'\n'<<"\033[0m" << std::flush;
        }


        // 循环推进
        current_token_id = next_token_id;

    }

    auto total_end_time = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time).count();

    std::cout << "\n\n---------------- 故事生成结束 ----------------\n";
    std::cout << "⏱️  总耗时: " << duration_ms / 1000.0f << " 秒\n";
    std::cout << "🚀 平均生成速度: " << (float)max_generate_step / (duration_ms / 1000.0f) << " tokens/秒\n";
}

// =========================================================================
// 🎮 绝对清爽的主程序入口
// =========================================================================
int main() {
    std::cout << "===================================================" << std::endl;
    std::cout << "🔥 HXInfer Engine - 纯 C++ 零依赖推理链路点火！🔥" << std::endl;
    std::cout << "===================================================" << std::endl;

    try {
        // 以后如果有新模型，比如 run_llama_7b()，直接在这里换一行代码就行了！
        run_llama_15m_stories();

    } catch (const std::exception& e) {
        std::cerr << "\n❌ 引擎崩溃: " << e.what() << std::endl;
    }

    return 0;
}