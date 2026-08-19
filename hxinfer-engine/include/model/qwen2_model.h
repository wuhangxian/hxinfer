#ifndef HXINFER_QWEN2_MODEL_H
#define HXINFER_QWEN2_MODEL_H
#include "model/causal_lm_model.h"
namespace hxinfer{

    // Qwen2 模型（Qwen2ForCausalLM）
    // 架构：RMSNorm + GQA Attention (with Q/K/V bias) + SwiGLU FFN
    // 兼容：Qwen2, Qwen2.5（权重命名相同，多了 bias）
    class Qwen2Model : public CausalLMModel {
    public:
        Qwen2Model(std::shared_ptr<Allocator> allocator,
                   std::shared_ptr<EmbeddingLayer>& embeddingLayer,
                   std::vector<std::shared_ptr<TransformerLayer>>& block,
                   std::shared_ptr<RMSNormLayer>& final_norm,
                   std::shared_ptr<LinearLayer>& lm_head,
                   ModelConfig& config)
            : CausalLMModel(allocator, embeddingLayer, block, final_norm, lm_head, config) {
            layer_name_ = "Qwen2Model";
        }
    };
}
#endif //HXINFER_QWEN2_MODEL_H
