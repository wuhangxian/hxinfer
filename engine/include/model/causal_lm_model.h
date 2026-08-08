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
        DataType activation_dtype_;  // 激活值类型：7B=FP16, 15M=FP32
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
            // 激活值类型：根据第一个 block 的权重类型决定
            activation_dtype_ = (!blocks_.empty() &&
                blocks_[0]->get_attention()->get_k_cache()->tensor_data_type() == DataType::kDataTypeFP16)
                ? DataType::kDataTypeFP16 : DataType::kDataTypeFP32;

            std::vector<int> hidden_shape={1,dim};
            ping_=std::make_shared<Tensor>(allocator,hidden_shape,activation_dtype_);
            pang_=std::make_shared<Tensor>(allocator,hidden_shape,activation_dtype_);
            DeviceType dev=allocator->device_type();
            ping_->tensor_set_device_type(dev);
            pang_->tensor_set_device_type(dev);
        }
        void forward(std::shared_ptr<Tensor>& input,std::shared_ptr<Tensor>& output,int pos);

        std::vector<std::shared_ptr<TransformerLayer>>& get_blocks() { return blocks_; }

        void forward(std::shared_ptr<Tensor>& input,std::shared_ptr<Tensor>& output) override{
            throw std::runtime_error("LlamaModel 需要pos参数!");
        }
    };
}
#endif //HXINFER_LLAMA_MODEL_H
