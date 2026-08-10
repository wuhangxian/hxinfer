#ifndef HXINFER_CAUSAL_LM_MODEL_H
#define HXINFER_CAUSAL_LM_MODEL_H
#include "layer/layer.h"
#include "layer/embedding.h"
#include "layer/transformer.h"
#include "layer/linear.h"
namespace hxinfer{

    class CausalLMModel:public Layer{
    protected:
        std::shared_ptr<EmbeddingLayer> embeddingLayer_;
        std::vector<std::shared_ptr<TransformerLayer>> blocks_;
        std::shared_ptr<RMSNormLayer> final_norm_;
        std::shared_ptr<LinearLayer> lm_head_;

        std::shared_ptr<Tensor> ping_;
        std::shared_ptr<Tensor> pang_;
        std::shared_ptr<Tensor> prefill_ping_;
        std::shared_ptr<Tensor> prefill_pang_;

        ModelConfig& config_;
        DataType activation_dtype_;

    public:
        CausalLMModel(std::shared_ptr<Allocator> allocator,
                   std::shared_ptr<EmbeddingLayer>& embeddingLayer,
                   std::vector<std::shared_ptr<TransformerLayer>>& block,
                   std::shared_ptr<RMSNormLayer>& final_norm,
                   std::shared_ptr<LinearLayer>& lm_head,
                   ModelConfig& config):Layer("CausalLMModel"),
                   embeddingLayer_(embeddingLayer),
                   blocks_(block),final_norm_(final_norm),lm_head_(lm_head),
                   config_(config){
            int dim=config_.dim;
            int max_seq=config_.seq_len;
            activation_dtype_ = (!blocks_.empty() &&
                blocks_[0]->get_attention()->get_k_cache()->tensor_data_type() == DataType::kDataTypeFP16)
                ? DataType::kDataTypeFP16 : DataType::kDataTypeFP32;
            std::vector<int> hidden_shape={1,dim};
            ping_=std::make_shared<Tensor>(allocator,hidden_shape,activation_dtype_);
            pang_=std::make_shared<Tensor>(allocator,hidden_shape,activation_dtype_);
            std::vector<int> prefill_shape={max_seq,dim};
            prefill_ping_=std::make_shared<Tensor>(allocator,prefill_shape,activation_dtype_);
            prefill_pang_=std::make_shared<Tensor>(allocator,prefill_shape,activation_dtype_);
            DeviceType dev=allocator->device_type();
            ping_->tensor_set_device_type(dev);
            pang_->tensor_set_device_type(dev);
            prefill_ping_->tensor_set_device_type(dev);
            prefill_pang_->tensor_set_device_type(dev);
        }

        void forward(std::shared_ptr<Tensor>& input,std::shared_ptr<Tensor>& output,int pos);
        void forward_prefill(std::shared_ptr<Tensor>& input_ids, std::shared_ptr<Tensor>& output, int seq_len);

        std::vector<std::shared_ptr<TransformerLayer>>& get_blocks() { return blocks_; }

        void forward(std::shared_ptr<Tensor>& input,std::shared_ptr<Tensor>& output) override{
            throw std::runtime_error("CausalLMModel requires 'pos' parameter!");
        }
    };
}
#endif
