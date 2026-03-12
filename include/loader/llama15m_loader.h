#ifndef HXINFER_LLAMA15M_LOADER_H
#define HXINFER_LLAMA15M_LOADER_H
#include "model/llama_model.h"
namespace hxinfer{
    class Llama15MLoader{
    public:
        static std::shared_ptr<LlamaModel> load_model(
            const std::string& model_path,
            ModelConfig& out_config,
            const std::shared_ptr<Allocator> allocator);

    };
}

#endif //HXINFER_LLAMA15M_LOADER_H
