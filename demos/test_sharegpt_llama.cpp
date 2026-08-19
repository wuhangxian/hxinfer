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
#include "llama_weight_loader.h"
#include "llama7b_tokenizer.h"

using namespace hxinfer;
using json = nlohmann::json;

int main(int argc, char** argv) {
    try {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    const char* env_data = std::getenv("HXINFER_DATA_DIR");
    std::string DATA_DIR = env_data ? std::string(env_data) : "/data/models/Llama-2-7b-hf";
    std::string TOKEN_PATH = DATA_DIR + "/tokenizer.model";
    std::string INPUT_JSON = "/data/dorianwu/datasets/sharegpt_100.json";
    std::string OUTPUT_JSON = "/data/dorianwu/datasets/sharegpt_llama_results.json";
    int MAX_NEW_TOKENS = 64;
    int MAX_PROMPT_TOKENS = 512;  // leave room for generation within seq_len=4096

    std::cout << ">>> Loading model from " << DATA_DIR << " ...\n";
    auto cpu_alloc  = std::make_shared<CPUAllocator>();
    auto cuda_alloc = std::make_shared<CUDAAllocator>();
    ModelConfig config;
    auto model = LlamaWeightLoader::load(DATA_DIR, config, cpu_alloc, cuda_alloc);

    std::cout << ">>> Loading tokenizer...\n";
    Llama7BTokenizer tokenizer(TOKEN_PATH);

    auto input_cpu = std::make_shared<Tensor>(cpu_alloc, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto input_gpu = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1}, DataType::kDataTypeFP32);
    auto logits    = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1, config.vocab_size}, DataType::kDataTypeFP32);
    input_gpu->tensor_set_device_type(DeviceType::kDeviceCUDA);
    logits->tensor_set_device_type(DeviceType::kDeviceCUDA);

    // Load ShareGPT prompts
    std::ifstream ifs(INPUT_JSON);
    if (!ifs) { std::cerr << "Cannot open " << INPUT_JSON << "\n"; return 1; }
    std::stringstream ss;
    ss << ifs.rdbuf();
    ifs.close();
    std::string raw = ss.str();
    std::cout << "Raw JSON size: " << raw.size() << " bytes\n";
    json prompts_data = json::parse(raw);
    std::cout << "Loaded " << prompts_data.size() << " prompts\n";
    std::cout << "Type of prompts_data: " << prompts_data.type_name() << "\n";
    if (!prompts_data.empty()) {
        std::cout << "First element type: " << prompts_data[0].type_name() << "\n";
        std::cout << "First element keys: ";
        for (auto& el : prompts_data[0].items()) std::cout << el.key() << " ";
        std::cout << "\n";
        std::cout << "prompt type: " << prompts_data[0]["prompt"].type_name() << "\n";
    }

    json results = json::array();

    for (int conv_idx = 0; conv_idx < (int)prompts_data.size(); conv_idx++) {
        std::string prompt = prompts_data[conv_idx]["prompt"].get<std::string>();
        std::string orig_id = prompts_data[conv_idx].value("orig_id", std::to_string(conv_idx));

        // Truncate long prompts
        if (prompt.length() > 4000) prompt = prompt.substr(0, 4000);

        std::vector<int> tokens = tokenizer.encode(prompt);
        tokens.insert(tokens.begin(), tokenizer.bos_id());

        // Truncate prompt tokens
        if ((int)tokens.size() > MAX_PROMPT_TOKENS) {
            tokens.resize(MAX_PROMPT_TOKENS);
        }

        // Prefill
        int next_token = tokens[0];
        for (int pos = 0; pos < (int)tokens.size(); pos++) {
            int* ptr = input_cpu->tensor_data_ptr<int>();
            ptr[0] = tokens[pos];
            cudaMemcpy(input_gpu->raw_data_ptr(), input_cpu->raw_data_ptr(),
                       sizeof(int), cudaMemcpyHostToDevice);
            model->forward(input_gpu, logits, pos);
            if (pos == (int)tokens.size() - 1)
                next_token = argmax_tensor(logits);
        }

        // Decode
        std::string generated_text;
        std::vector<int> generated_token_ids;
        int pos = (int)tokens.size();
        int generated = 0;
        while (generated < MAX_NEW_TOKENS) {
            if (next_token == tokenizer.eos_id()) break;
            std::string word = tokenizer.decode(next_token);
            generated_text += word;
            generated_token_ids.push_back(next_token);

            int* ptr = input_cpu->tensor_data_ptr<int>();
            ptr[0] = next_token;
            cudaMemcpy(input_gpu->raw_data_ptr(), input_cpu->raw_data_ptr(),
                       sizeof(int), cudaMemcpyHostToDevice);
            model->forward(input_gpu, logits, pos);
            next_token = argmax_tensor(logits);
            pos++;
            generated++;
        }

        // Check for garbled output (high ratio of non-ASCII / control chars)
        int non_ascii = 0;
        int control = 0;
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
            {"orig_id", orig_id},
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

    json summary = {
        {"total", results.size()},
        {"garbled_count", garbled_count},
        {"results", results}
    };

    std::ofstream ofs(OUTPUT_JSON);
    ofs << summary.dump(2, ' ', false, json::error_handler_t::replace);

    std::cout << "\n=== SUMMARY ===\n";
    std::cout << "Total: " << results.size() << "\n";
    std::cout << "Garbled: " << garbled_count << "\n";
    std::cout << "Output saved to " << OUTPUT_JSON << "\n";

    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
