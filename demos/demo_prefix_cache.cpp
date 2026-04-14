// =========================================================================
// demo_prefix_cache.cpp — Prefix Cache 功能演示
//
// 本 Demo 演示了 PrefixCacheManager 的核心价值：
//   对于相同的 token 前缀，第二次推理可以跳过前缀部分的 forward 计算，
//   直接从缓存中恢复 KV Cache，从而节省时间。
//
// 流程：
//   Round 1 (冷启动): 从 BOS 开始跑完整 200 步 → 计时 T_cold
//                      → 记录生成的所有 token
//                      → 快照前 N 个位置的 KV Cache 到 PrefixCacheManager
//
//   Round 2 (热启动): 从 PrefixCacheManager 恢复前 N 行 KV Cache
//                      → 直接从 pos=N 开始，只跑 200-N 步 → 计时 T_warm
//
//   验证: Round 2 从 pos=N 开始的输出应与 Round 1 完全一致（贪心解码确定性）
//
// 编译: cmake --build build --target demo_prefix_cache
// 运行: cd /home/whx/hxinfer && ./build/demo_prefix_cache
// =========================================================================

#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <string>
#include <cassert>

#include "base/allocator.h"
#include "base/config.h"
#include "tensor/tensor.h"
#include "op/math_ops.h"
#include "model/llama_model.h"
#include "loader/llama15m_loader.h"
#include "loader/llama_tokenizer.h"
#include "cache/prefix_cache.h"

using namespace hxinfer;

int main() {
    std::cout << "===========================================================\n";
    std::cout << "   HXInfer — Prefix Cache Demo (CPU)\n";
    std::cout << "===========================================================\n";

    // =====================================================================
    // 第 0 步：加载模型和分词器（只加载一次，两轮共用）
    // =====================================================================
    auto cpu_allocator = std::make_shared<CPUAllocator>();

    std::string model_path = "models/stories15M.bin";
    std::string tokenizer_path = "models/tokenizer.bin";
    ModelConfig config;

    std::cout << "\n>>> 加载模型..." << std::endl;
    auto llama_engine = Llama15MLoader::load_model(model_path, config, cpu_allocator);

    std::cout << ">>> 加载分词器..." << std::endl;
    LlamaTokenizer tokenizer(tokenizer_path, config.vocab_size);

    // 准备 I/O Tensor（两轮共用）
    auto input_tensor = std::make_shared<Tensor>(
        cpu_allocator, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto logits_tensor = std::make_shared<Tensor>(
        cpu_allocator, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP32);

    // =====================================================================
    // 配置参数
    // =====================================================================
    const int TOTAL_STEPS = 200;    // 总生成步数
    const int PREFIX_LEN  = 20;     // 前缀长度（前 20 个 token 会被缓存）

    std::cout << "\n>>> 配置: 总步数=" << TOTAL_STEPS
              << ", 前缀长度=" << PREFIX_LEN << "\n";

    // =====================================================================
    // Round 1：冷启动 — 完整跑 200 步，记录所有 token
    //
    // 目的：
    //   1. 完成一次完整的推理，作为 baseline 计时
    //   2. 记录下所有生成的 token（用于 Round 2 的验证和前缀恢复）
    //   3. 跑完后从模型的 KV Cache 中快照前 PREFIX_LEN 行
    // =====================================================================
    std::cout << "\n==================== Round 1: 冷启动 ====================\n";

    // generated_tokens[i] = 第 i 步生成的 next token id
    // 特别注意：generated_tokens[0] 是 pos=0 时 forward 的输出
    //          而 pos=0 时的输入是 BOS (token id = 1)
    std::vector<int> generated_tokens;
    generated_tokens.reserve(TOTAL_STEPS);

    int current_token = 1; // BOS (Begin of Sequence)

    auto t1_start = std::chrono::high_resolution_clock::now();

    for (int pos = 0; pos < TOTAL_STEPS; pos++) {
        // 把当前 token id 写入 input tensor
        int* ptr = input_tensor->tensor_data_ptr<int>();
        ptr[0] = current_token;

        // forward：数据穿透 6 层 Transformer
        llama_engine->forward(input_tensor, logits_tensor, pos);

        // argmax 贪心解码：从 32000 个 logits 中找最大的
        int next_token = argmax_tensor(logits_tensor);
        generated_tokens.push_back(next_token);

        // EOS 检查
        if (next_token == 2) break;

        current_token = next_token;
    }

    auto t1_end = std::chrono::high_resolution_clock::now();
    float t1_ms = std::chrono::duration_cast<std::chrono::microseconds>(
        t1_end - t1_start).count() / 1000.0f;

    // 打印 Round 1 生成的文本
    std::cout << "\n[Round 1 输出] ";
    for (int tok : generated_tokens) {
        std::cout << tokenizer.decode(tok);
    }
    std::cout << "\n\nRound 1 耗时: " << t1_ms << " ms ("
              << generated_tokens.size() << " steps, "
              << generated_tokens.size() / (t1_ms / 1000.0f) << " tok/s)\n";

    // =====================================================================
    // 快照阶段：把前 PREFIX_LEN 个位置的 KV Cache 存入 PrefixCacheManager
    //
    // 此时模型的 KV Cache 第 0~199 行都已被填充。
    // 但我们只需要前 PREFIX_LEN=20 行——因为这些行的值只取决于前 20 个 token，
    // 后续的 token 不会修改它们（KV Cache 的每一行写入后永远不变）。
    //
    // prefix_tokens 的含义：
    //   我们需要存储的是"导致这 20 行 KV Cache 被写入的那 20 个输入 token"。
    //   pos=0 的输入是 BOS(1)，pos=1 的输入是 generated_tokens[0]，...
    //   pos=19 的输入是 generated_tokens[18]。
    //   所以 prefix_tokens = [BOS] + generated_tokens[0..18]，共 20 个。
    // =====================================================================
    std::cout << "\n==================== 快照 KV Cache ====================\n";

    // 构造 prefix token 序列
    // prefix_tokens[0] = BOS = 1
    // prefix_tokens[1] = generated_tokens[0] (pos=0 输出的 token)
    // prefix_tokens[2] = generated_tokens[1] (pos=1 输出的 token)
    // ...
    // prefix_tokens[19] = generated_tokens[18]
    std::vector<int> prefix_tokens;
    prefix_tokens.push_back(1); // BOS
    for (int i = 0; i < PREFIX_LEN - 1 && i < (int)generated_tokens.size(); i++) {
        prefix_tokens.push_back(generated_tokens[i]);
    }

    // 打印前缀内容，方便验证
    std::cout << "[前缀 tokens] ";
    for (int tok : prefix_tokens) {
        std::cout << tokenizer.decode(tok);
    }
    std::cout << " (" << prefix_tokens.size() << " tokens)\n";

    // 创建 PrefixCacheManager 并存入快照
    PrefixCacheManager cache_manager(cpu_allocator, config);
    cache_manager.store(prefix_tokens, *llama_engine);

    // =====================================================================
    // Round 2：热启动 — 恢复 KV Cache，从 pos=PREFIX_LEN 开始
    //
    // 核心逻辑：
    //   1. cache_manager.restore() 把快照 memcpy 回模型的 KV Cache 前 N 行
    //      → 此时 KV Cache[0..N-1] 和 Round 1 完全一致
    //   2. 从 pos=N 开始 forward，输入的第一个 token = generated_tokens[N-1]
    //      （因为 pos=N 的输入 = pos=N-1 的输出）
    //   3. attention 计算时会从 cache[0..pos] 读取，其中 cache[0..N-1] 来自快照，
    //      cache[N..pos] 来自本轮 forward 的实时写入
    //   4. 由于贪心解码是确定性的，Round 2 的输出应该和 Round 1 完全一致
    //
    // 为什么不需要"重置"KV Cache？
    //   cache[N..255] 里可能有 Round 1 的旧数据，但无所谓——
    //   因为 pos=N 时 attention 只看 cache[0..N]，不会看 N+1 以后的位置。
    //   之后每步 forward 会把新的 K/V 写入 cache[pos]，覆盖旧数据。
    // =====================================================================
    std::cout << "\n==================== Round 2: 热启动 ====================\n";

    // Step 1: 查找缓存
    int matched = cache_manager.lookup(prefix_tokens);
    std::cout << "[PrefixCache] lookup 结果: "
              << (matched > 0 ? "命中！" : "未命中")
              << " prefix_len=" << matched << "\n";

    // Step 2: 恢复 KV Cache
    auto restore_start = std::chrono::high_resolution_clock::now();
    cache_manager.restore(prefix_tokens, *llama_engine);
    auto restore_end = std::chrono::high_resolution_clock::now();
    float restore_us = std::chrono::duration_cast<std::chrono::microseconds>(
        restore_end - restore_start).count();

    std::cout << "[PrefixCache] restore 耗时: " << restore_us << " us\n";

    // Step 3: 从 pos=PREFIX_LEN 开始生成
    // 输入的第一个 token = generated_tokens[PREFIX_LEN - 1]
    // 因为 pos=PREFIX_LEN 时，输入应该是 pos=PREFIX_LEN-1 输出的 token
    std::vector<int> round2_tokens;
    round2_tokens.reserve(TOTAL_STEPS - PREFIX_LEN);

    current_token = generated_tokens[PREFIX_LEN - 1];

    auto t2_start = std::chrono::high_resolution_clock::now();

    for (int pos = PREFIX_LEN; pos < TOTAL_STEPS; pos++) {
        int* ptr = input_tensor->tensor_data_ptr<int>();
        ptr[0] = current_token;

        llama_engine->forward(input_tensor, logits_tensor, pos);

        int next_token = argmax_tensor(logits_tensor);
        round2_tokens.push_back(next_token);

        if (next_token == 2) break;

        current_token = next_token;
    }

    auto t2_end = std::chrono::high_resolution_clock::now();
    float t2_ms = std::chrono::duration_cast<std::chrono::microseconds>(
        t2_end - t2_start).count() / 1000.0f;

    // 打印 Round 2 输出
    std::cout << "\n[Round 2 输出] ";
    // 先打印前缀部分（来自缓存，没有实际 forward）
    for (int i = 0; i < PREFIX_LEN && i < (int)generated_tokens.size(); i++) {
        std::cout << tokenizer.decode(generated_tokens[i]);
    }
    // 再打印本轮实际生成的部分
    for (int tok : round2_tokens) {
        std::cout << tokenizer.decode(tok);
    }
    std::cout << "\n\nRound 2 耗时: " << t2_ms << " ms ("
              << round2_tokens.size() << " decode steps, "
              << round2_tokens.size() / (t2_ms / 1000.0f) << " tok/s)\n";

    // =====================================================================
    // 验证 + 对比
    //
    // 验证：Round 2 生成的每个 token 应该和 Round 1 对应位置完全一致
    //       因为 KV Cache 相同 + 输入 token 相同 + 贪心解码 → 输出必然相同
    //
    // 对比：Round 2 少跑了 PREFIX_LEN 步 forward，应该更快
    // =====================================================================
    std::cout << "\n==================== 结果对比 ====================\n";

    // 验证一致性
    bool match = true;
    for (int i = 0; i < (int)round2_tokens.size(); i++) {
        int round1_idx = PREFIX_LEN + i; // Round 1 中对应的位置
        if (round1_idx < (int)generated_tokens.size()) {
            if (round2_tokens[i] != generated_tokens[round1_idx]) {
                std::cout << "[MISMATCH] pos=" << (PREFIX_LEN + i)
                          << " Round1=" << generated_tokens[round1_idx]
                          << " Round2=" << round2_tokens[i] << "\n";
                match = false;
            }
        }
    }

    std::cout << "输出一致性验证: " << (match ? "PASS" : "FAIL") << "\n\n";

    // 性能对比
    float speedup = (t1_ms - t2_ms) / t1_ms * 100.0f;
    std::cout << "Round 1 (冷启动): " << t1_ms << " ms, "
              << generated_tokens.size() << " steps\n";
    std::cout << "Round 2 (热启动): " << t2_ms << " ms, "
              << round2_tokens.size() << " steps"
              << " + restore " << restore_us << " us\n";
    std::cout << "跳过的 forward 步数: " << PREFIX_LEN << "\n";
    std::cout << "节省时间: " << (t1_ms - t2_ms) << " ms ("
              << speedup << "%)\n";

    return 0;
}
