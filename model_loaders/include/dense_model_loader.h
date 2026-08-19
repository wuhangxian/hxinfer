#ifndef HXINFER_DENSE_MODEL_LOADER_H
#define HXINFER_DENSE_MODEL_LOADER_H
#include "model/causal_lm_model.h"
#include "safetensors_loader.h"
#include "base/allocator.h"
#include <string>
#include <memory>

namespace hxinfer {

enum class ModelType {
    LLaMA,   // LLaMA-2/3, Mistral, InternLM2/3 (no QKV bias)
    Qwen2,   // Qwen2/2.5 (has QKV bias)
};

class DenseModelLoader {
public:
    static std::shared_ptr<CausalLMModel> load(
        const std::string& model_dir,
        ModelConfig& out_config,
        const std::shared_ptr<CPUAllocator>& cpu_alloc,
        const std::shared_ptr<CUDAAllocator>& cuda_alloc,
        ModelType type = ModelType::LLaMA
    );
};

} // namespace hxinfer
#endif
