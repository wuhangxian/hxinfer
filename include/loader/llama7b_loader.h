#ifndef HXINFER_LLAMA7B_LOADER_H
#define HXINFER_LLAMA7B_LOADER_H
#include "model/llama_model.h"
#include "base/allocator.h"
#include <string>
#include <memory>

namespace hxinfer{
    class Llama7BLoader{
    public:
        // data_dir: /workspace/whx/hxinfer-data
        // 读取 data_dir/Yarn-Llama-2-7b-128k/config.json
        //      data_dir/weights/llama7b_index.json
        //      data_dir/weights/llama7b_weights.bin
        static std::shared_ptr<LlamaModel> load(
            const std::string& data_dir,
            ModelConfig& out_config,
            const std::shared_ptr<CPUAllocator>& cpu_alloc,
            const std::shared_ptr<CUDAAllocator>& cuda_alloc
        );
    };
}
#endif
