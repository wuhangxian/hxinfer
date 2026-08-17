#ifndef HXINFER_LLAMA_MODEL_H
#define HXINFER_LLAMA_MODEL_H
#include "model/causal_lm_model.h"
namespace hxinfer{

    // LLaMA-2 模型（LlamaForCausalLM）
    // 架构：RMSNorm + MHA/GQA Attention + SwiGLU FFN
    // 兼容：LLaMA-2, LLaMA-3, Mistral, InternLM2/3（权重命名相同）
    class LlamaModel : public CausalLMModel {
    public:
        LlamaModel(std::shared_ptr<Allocator> allocator,
                   std::shared_ptr<EmbeddingLayer>& embeddingLayer,
                   std::vector<std::shared_ptr<TransformerLayer>>& block,
                   std::shared_ptr<RMSNormLayer>& final_norm,
                   std::shared_ptr<LinearLayer>& lm_head,
                   ModelConfig& config)
            : CausalLMModel(allocator, embeddingLayer, block, final_norm, lm_head, config) {
            layer_name_ = "LlamaModel";
        }
    };
}
#endif //HXINFER_LLAMA_MODEL_H
