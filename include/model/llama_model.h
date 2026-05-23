#ifndef HXINFER_LLAMA_MODEL_H
#define HXINFER_LLAMA_MODEL_H
#include "layer/layer.h"
#include "layer/embedding.h"
#include "layer/transformer.h"
#include "layer/linear.h"
namespace hxinfer{
    class LlamaModel:public Layer{
    private:
        std::shared_ptr<EmbeddingLayer> embeddingLayer_;
        std::vector<std::shared_ptr<TransformerLayer>> blocks_;
        std::shared_ptr<RMSNormLayer> final_norm_;
        std::shared_ptr<LinearLayer> lm_head_;
        std::shared_ptr<Tensor> ping_;
        std::shared_ptr<Tensor> pang_;

        ModelConfig& config_;
    public:
        LlamaModel(std::shared_ptr<Allocator> allocator,
                   std::shared_ptr<EmbeddingLayer>& embeddingLayer,
                   std::vector<std::shared_ptr<TransformerLayer>>& block,
                   std::shared_ptr<RMSNormLayer>& final_norm,
                   std::shared_ptr<LinearLayer>& lm_head,
                   ModelConfig& config):Layer("LlamaModel"),
                   embeddingLayer_(embeddingLayer),
                   blocks_(block),final_norm_(final_norm),lm_head_(lm_head),
                   config_(config){
            int dim=config_.dim;
            std::vector<int> hidden_shape={1,dim};
            DataType dtype=DataType::kDataTypeFP32;
            ping_=std::make_shared<Tensor>(allocator,hidden_shape,dtype);
            pang_=std::make_shared<Tensor>(allocator,hidden_shape,dtype);
        }
        void forward(std::shared_ptr<Tensor>& input,std::shared_ptr<Tensor>& output,int pos);

        // ===================== Prefix Cache 专用接口 =====================
        // 为什么要暴露 blocks_？
        // PrefixCacheManager 需要遍历模型的每一层 TransformerLayer，
        // 然后通过 get_attention() → get_k/v_cache() 拿到每层的 KV Cache。
        // 调用链: model.get_blocks()[i] → blocks_[i]->get_attention() → get_k/v_cache()
        std::vector<std::shared_ptr<TransformerLayer>>& get_blocks() { return blocks_; }

        void forward(std::shared_ptr<Tensor>& input,std::shared_ptr<Tensor>& output) override{
            throw std::runtime_error("LlamaModel 需要pos参数!");
        }
    };
}
#endif //HXINFER_LLAMA_MODEL_H
