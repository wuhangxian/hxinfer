#ifndef HXINFER_INFERENCE_ENGINE_H
#define HXINFER_INFERENCE_ENGINE_H
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include "base/allocator.h"
#include "base/config.h"
#include "tensor/tensor.h"
#include "op/math_ops.h"
#include "dense_model_loader.h"

namespace hxinfer {

class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();

    void load(const std::string& model_dir, ModelType type,
              const std::string& tokenizer_path,
              const std::string& tokenizer_type = "llama");

    struct GenerateResult {
        std::string text;
        std::vector<int> token_ids;
        int prompt_tokens;
        int generated_tokens;
        double prefill_time_ms;
        double decode_time_ms;
        double total_time_ms;
    };

    GenerateResult generate(const std::string& prompt, int max_new_tokens = 128,
                            float temperature = 0.0f, float top_p = 1.0f);

    bool is_loaded() const { return model_ != nullptr; }

private:
    std::shared_ptr<CPUAllocator> cpu_alloc_;
    std::shared_ptr<CUDAAllocator> cuda_alloc_;
    ModelConfig config_;
    std::shared_ptr<CausalLMModel> model_;

    // Tokenizer — we keep both and use the right one based on model type
    void* tokenizer_ = nullptr;  // points to either Llama7BTokenizer or QwenTokenizer
    int tokenizer_type_ = 0;     // 0=llama, 1=qwen
    std::string bos_id_str_;

    // Reusable I/O tensors
    std::shared_ptr<Tensor> input_cpu_;
    std::shared_ptr<Tensor> input_gpu_;
    std::shared_ptr<Tensor> logits_;

    std::mutex mutex_;

    // Tokenizer interface (type-erased)
    std::vector<int> encode(const std::string& text);
    std::string decode(int token_id);
    int eos_id();
    int bos_id();
};

} // namespace hxinfer
#endif
