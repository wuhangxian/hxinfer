#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <vector>
#include <chrono>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <nlohmann/json.hpp>
#include "cuda_runtime.h"
#include "base/allocator.h"
#include "base/config.h"
#include "tensor/tensor.h"
#include "op/math_ops.h"
#include "dense_model_loader.h"
#include "qwen_tokenizer.h"

using namespace hxinfer;
using json = nlohmann::json;

static std::vector<int> apply_chat_template(QwenTokenizer& tok, const std::string& user_msg) {
    std::string chat = "<|im_start|>user\n" + user_msg + "<|im_end|>\n<|im_start|>assistant\n";
    return tok.encode(chat);
}

int main() {
    try {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    std::string DATA_DIR = "/data/models/Qwen2.5-7B-Instruct";
    std::string TOKEN_PATH = DATA_DIR + "/tokenizer.json";
    std::string INPUT_JSON = "/data/dorianwu/datasets/sharegpt_100.json";
    std::string OUTPUT_JSON = "/data/dorianwu/datasets/sharegpt_qwen_results.json";
    int MAX_NEW_TOKENS = 64;
    int MAX_PROMPT_TOKENS = 512;

    std::cout << ">>> Loading Qwen2.5-7B from " << DATA_DIR << " ...\n";
    auto cpu_alloc  = std::make_shared<CPUAllocator>();
    auto cuda_alloc = std::make_shared<CUDAAllocator>();
    ModelConfig config;
    auto model = DenseModelLoader::load(DATA_DIR, config, cpu_alloc, cuda_alloc, ModelType::Qwen2);

    std::cout << ">>> Loading tokenizer...\n";
    QwenTokenizer tokenizer(TOKEN_PATH);

    auto input_cpu = std::make_shared<Tensor>(cpu_alloc, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto input_gpu = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto logits    = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP32);
    input_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
    logits->tensor_set_device_type(DeviceType::kDeviceCUDA);

    std::ifstream ifs(INPUT_JSON);
    if (!ifs) { std::cerr << "Cannot open " << INPUT_JSON << "\n"; return 1; }
    json prompts_data = json::parse(ifs);

    json results = json::array();

    for (int conv_idx = 0; conv_idx < (int)prompts_data.size(); conv_idx++) {
        std::string prompt = prompts_data[conv_idx]["prompt"].get<std::string>();
        if (prompt.length() > 4000) prompt = prompt.substr(0, 4000);

        std::vector<int> tokens = apply_chat_template(tokenizer, prompt);
        if ((int)tokens.size() > MAX_PROMPT_TOKENS) tokens.resize(MAX_PROMPT_TOKENS);

        int next_token = tokens[0];
        for (int pos = 0; pos < (int)tokens.size(); pos++) {
            input_cpu->tensor_data_ptr<int>()[0] = tokens[pos];
            cudaMemcpy(input_gpu->raw_data_ptr(), input_cpu->raw_data_ptr(), sizeof(int), cudaMemcpyHostToDevice);
            model->forward(input_gpu, logits, pos);
            if (pos == (int)tokens.size() - 1) next_token = argmax_tensor(logits);
        }

        std::string generated_text;
        std::vector<int> generated_token_ids;
        int pos = (int)tokens.size();
        int generated = 0;
        while (generated < MAX_NEW_TOKENS) {
            if (next_token == tokenizer.eos_id()) break;
            std::string word = tokenizer.decode(next_token);
            generated_text += word;
            generated_token_ids.push_back(next_token);
            input_cpu->tensor_data_ptr<int>()[0] = next_token;
            cudaMemcpy(input_gpu->raw_data_ptr(), input_cpu->raw_data_ptr(), sizeof(int), cudaMemcpyHostToDevice);
            model->forward(input_gpu, logits, pos);
            next_token = argmax_tensor(logits);
            pos++;
            generated++;
        }

        int non_ascii = 0, control = 0;
        for (char c : generated_text) {
            if ((unsigned char)c > 127) non_ascii++;
            if ((unsigned char)c < 32 && c != '\n' && c != '\t' && c != '\r') control++;
        }
        bool garbled = (generated_text.length() > 0 &&
                        (non_ascii > (int)generated_text.length() * 0.3 ||
                         control > (int)generated_text.length() * 0.2));

        std::cout << "[" << (conv_idx+1) << "/" << prompts_data.size() << "]"
                  << " prompt_tokens=" << tokens.size()
                  << " generated_tokens=" << generated
                  << " garbled=" << (garbled ? "YES" : "no")
                  << " output_preview=" << generated_text.substr(0, 80)
                  << "\n";

        results.push_back({
            {"id", conv_idx},
            {"prompt", prompt.substr(0, 300)},
            {"prompt_tokens", (int)tokens.size()},
            {"generated_tokens", generated},
            {"output", generated_text.substr(0, 500)},
            {"token_ids", generated_token_ids},
            {"garbled", garbled}
        });
    }

    int garbled_count = 0;
    for (auto& r : results) if (r["garbled"].get<bool>()) garbled_count++;

    json summary = {{"total", results.size()}, {"garbled_count", garbled_count}, {"results", results}};
    std::ofstream ofs(OUTPUT_JSON);
    ofs << summary.dump(2, ' ', false, json::error_handler_t::replace);

    std::cout << "\n=== SUMMARY ===\nTotal: " << results.size() << "\nGarbled: " << garbled_count << "\n";
    std::cout << "Output saved to " << OUTPUT_JSON << "\n";
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
