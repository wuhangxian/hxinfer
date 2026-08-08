#ifndef HXINFER_LLAMA_WEIGHT_LOADER_H
#define HXINFER_LLAMA_WEIGHT_LOADER_H

#include "model/llama_model.h"
#include "safetensors_loader.h"
#include "base/allocator.h"
#include <string>
#include <memory>

namespace hxinfer {

class LlamaWeightLoader {
public:
    static std::shared_ptr<LlamaModel> load(
        const std::string& model_dir,
        ModelConfig& out_config,
        const std::shared_ptr<CPUAllocator>& cpu_alloc,
        const std::shared_ptr<CUDAAllocator>& cuda_alloc
    );
};

} // namespace hxinfer
#endif // HXINFER_LLAMA_WEIGHT_LOADER_H
