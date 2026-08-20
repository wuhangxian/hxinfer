#include "inference_engine.h"
#include "llama7b_tokenizer.h"
#include "qwen_tokenizer.h"
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

namespace hxinfer {

InferenceEngine::InferenceEngine() {
    cpu_alloc_ = std::make_shared<CPUAllocator>();
    cuda_alloc_ = std::make_shared<CUDAAllocator>();
}

InferenceEngine::~InferenceEngine() {
    if (tokenizer_) {
        if (tokenizer_type_ == 0) delete static_cast<Llama7BTokenizer*>(tokenizer_);
        else delete static_cast<QwenTokenizer*>(tokenizer_);
    }
}

void InferenceEngine::load(const std::string& model_dir, ModelType type,
                          const std::string& tokenizer_path,
                          const std::string& tokenizer_type) {
    model_ = DenseModelLoader::load(model_dir, config_, cpu_alloc_, cuda_alloc_, type);

    if (tokenizer_type == "qwen" || type == ModelType::Qwen2) {
        tokenizer_ = new QwenTokenizer(tokenizer_path);
        tokenizer_type_ = 1;
    } else {
        tokenizer_ = new Llama7BTokenizer(tokenizer_path);
        tokenizer_type_ = 0;
    }

    input_cpu_ = std::make_shared<Tensor>(cpu_alloc_, std::vector<int>{1}, DataType::kDataTypeFP32);
    input_gpu_ = std::make_shared<Tensor>(cuda_alloc_, std::vector<int>{1}, DataType::kDataTypeFP32);
    logits_ = std::make_shared<Tensor>(cuda_alloc_, std::vector<int>{1, config_.vocab_size}, config_.logits_dtype);
    input_gpu_->tensor_set_device_type(DeviceType::kDeviceCUDA);
    logits_->tensor_set_device_type(DeviceType::kDeviceCUDA);

    std::cout << "[InferenceEngine] Model loaded, ready to serve." << std::endl;
}

std::vector<int> InferenceEngine::encode(const std::string& text) {
    if (tokenizer_type_ == 0)
        return static_cast<Llama7BTokenizer*>(tokenizer_)->encode(text);
    else
        return static_cast<QwenTokenizer*>(tokenizer_)->encode(text);
}

std::string InferenceEngine::decode(int token_id) {
    if (tokenizer_type_ == 0)
        return static_cast<Llama7BTokenizer*>(tokenizer_)->decode(token_id);
    else
        return static_cast<QwenTokenizer*>(tokenizer_)->decode(token_id);
}

int InferenceEngine::eos_id() {
    if (tokenizer_type_ == 0)
        return static_cast<Llama7BTokenizer*>(tokenizer_)->eos_id();
    else
        return static_cast<QwenTokenizer*>(tokenizer_)->eos_id();
}

int InferenceEngine::bos_id() {
    if (tokenizer_type_ == 0)
        return static_cast<Llama7BTokenizer*>(tokenizer_)->bos_id();
    else
        return static_cast<QwenTokenizer*>(tokenizer_)->bos_id();
}

InferenceEngine::GenerateResult InferenceEngine::generate(
        const std::string& prompt, int max_new_tokens,
        float temperature, float top_p) {
    std::lock_guard<std::mutex> lock(mutex_);

    GenerateResult result;
    bool greedy = (temperature < 0.01f);

    // Tokenize
    std::vector<int> tokens = encode(prompt);
    int bos = bos_id();
    if (bos >= 0) tokens.insert(tokens.begin(), bos);
    result.prompt_tokens = tokens.size();

    // Prefill
    auto t_start = std::chrono::high_resolution_clock::now();
    int next_token = -1;
    for (int pos = 0; pos < (int)tokens.size(); pos++) {
        input_cpu_->tensor_data_ptr<int>()[0] = tokens[pos];
        cudaMemcpy(input_gpu_->raw_data_ptr(), input_cpu_->raw_data_ptr(),
                   sizeof(int), cudaMemcpyHostToDevice);
        model_->forward(input_gpu_, logits_, pos);
        if (pos == (int)tokens.size() - 1) {
            next_token = greedy ? argmax_tensor(logits_)
                                : sample_tensor(logits_, temperature, top_p);
        }
    }
    auto t_prefill_end = std::chrono::high_resolution_clock::now();
    result.prefill_time_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_start).count();

    // Decode
    int pos = (int)tokens.size();
    while (result.generated_tokens < max_new_tokens) {
        if (next_token == eos_id()) break;
        result.token_ids.push_back(next_token);
        result.text += decode(next_token);

        input_cpu_->tensor_data_ptr<int>()[0] = next_token;
        cudaMemcpy(input_gpu_->raw_data_ptr(), input_cpu_->raw_data_ptr(),
                   sizeof(int), cudaMemcpyHostToDevice);
        model_->forward(input_gpu_, logits_, pos);
        next_token = greedy ? argmax_tensor(logits_)
                            : sample_tensor(logits_, temperature, top_p);
        pos++;
        result.generated_tokens++;
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    result.decode_time_ms = std::chrono::duration<double, std::milli>(t_end - t_prefill_end).count();
    result.total_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    return result;
}

} // namespace hxinfer
