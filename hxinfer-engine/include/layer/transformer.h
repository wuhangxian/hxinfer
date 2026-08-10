#ifndef HXINFER_TRANSFORMER_H
#define HXINFER_TRANSFORMER_H
#include "layer.h"
#include "attention.h"
#include "swiglu.h"
#include "rmsnorm.h"
namespace hxinfer{
    class TransformerLayer:public Layer{
    private:
        std::shared_ptr<RMSNormLayer> attn_norm_;
        std::shared_ptr<AttentionLayer> attentionLayer_;
        std::shared_ptr<RMSNormLayer> ffn_norm_;
        std::shared_ptr<SwigluLayer> swigluLayer_;

        std::shared_ptr<Tensor> norm_out_;
        std::shared_ptr<Tensor> attn_out_;
        std::shared_ptr<Tensor> ffn_out_;

        // Prefill buffers
        std::shared_ptr<Tensor> prefill_norm_out_;
        std::shared_ptr<Tensor> prefill_attn_out_;
        std::shared_ptr<Tensor> prefill_ffn_out_;
        std::shared_ptr<Tensor> prefill_add_out_;

        ModelConfig config_;

    public:
        TransformerLayer(std::shared_ptr<Allocator> allocator,ModelConfig& modelConfig,
                         std::shared_ptr<Tensor>& attn_norm_w,std::shared_ptr<Tensor>& wq,
                         std::shared_ptr<Tensor>& wk,std::shared_ptr<Tensor>& wv,std::shared_ptr<Tensor>& wo,
                         std::shared_ptr<Tensor>& ffn_norm_w,std::shared_ptr<Tensor>& w_gate,
                         std::shared_ptr<Tensor>& w_up,std::shared_ptr<Tensor>& w_down,
                         bool enable_fused): Layer("Transformer"),
                         config_(modelConfig){
            int dim=config_.dim;
            int max_seq=config_.seq_len;
            DataType act_dtype = (wq->tensor_data_type() == DataType::kDataTypeFP16)
                                 ? DataType::kDataTypeFP16 : DataType::kDataTypeFP32;
            std::vector<int> buffer_shapes={1,dim};
            std::vector<int> prefill_shapes={max_seq,dim};
            norm_out_=std::make_shared<Tensor>(allocator,buffer_shapes,act_dtype);
            attn_out_=std::make_shared<Tensor>(allocator,buffer_shapes,act_dtype);
            ffn_out_=std::make_shared<Tensor>(allocator,buffer_shapes,act_dtype);
            prefill_norm_out_=std::make_shared<Tensor>(allocator,prefill_shapes,act_dtype);
            prefill_attn_out_=std::make_shared<Tensor>(allocator,prefill_shapes,act_dtype);
            prefill_ffn_out_=std::make_shared<Tensor>(allocator,prefill_shapes,act_dtype);
            prefill_add_out_=std::make_shared<Tensor>(allocator,prefill_shapes,act_dtype);
            DeviceType dev=allocator->device_type();
            norm_out_->tensor_set_device_type(dev);
            attn_out_->tensor_set_device_type(dev);
            ffn_out_->tensor_set_device_type(dev);
            prefill_norm_out_->tensor_set_device_type(dev);
            prefill_attn_out_->tensor_set_device_type(dev);
            prefill_ffn_out_->tensor_set_device_type(dev);
            prefill_add_out_->tensor_set_device_type(dev);
            attn_norm_=std::make_shared<RMSNormLayer>(attn_norm_w);
            attentionLayer_=std::make_shared<AttentionLayer>(allocator,config_,wq,wk,wv,wo);
            ffn_norm_=std::make_shared<RMSNormLayer>(ffn_norm_w);
            swigluLayer_=std::make_shared<SwigluLayer>(allocator,config_,w_gate,w_up,w_down);
            if(enable_fused) swigluLayer_->enable_fused();
        }

        TransformerLayer(std::shared_ptr<Allocator> allocator,ModelConfig& modelConfig,
                         std::shared_ptr<Tensor>& attn_norm_w,std::shared_ptr<Tensor>& wq,
                         std::shared_ptr<Tensor>& wk,std::shared_ptr<Tensor>& wv,std::shared_ptr<Tensor>& wo,
                         std::shared_ptr<Tensor>& ffn_norm_w,std::shared_ptr<Tensor>& w_gate,
                         std::shared_ptr<Tensor>& w_up,std::shared_ptr<Tensor>& w_down,
                         bool enable_fused,
                         std::shared_ptr<Tensor>& bq,std::shared_ptr<Tensor>& bk,std::shared_ptr<Tensor>& bv): Layer("Transformer"),
                         config_(modelConfig){
            int dim=config_.dim;
            int max_seq=config_.seq_len;
            DataType act_dtype = (wq->tensor_data_type() == DataType::kDataTypeFP16)
                                 ? DataType::kDataTypeFP16 : DataType::kDataTypeFP32;
            std::vector<int> buffer_shapes={1,dim};
            std::vector<int> prefill_shapes={max_seq,dim};
            norm_out_=std::make_shared<Tensor>(allocator,buffer_shapes,act_dtype);
            attn_out_=std::make_shared<Tensor>(allocator,buffer_shapes,act_dtype);
            ffn_out_=std::make_shared<Tensor>(allocator,buffer_shapes,act_dtype);
            prefill_norm_out_=std::make_shared<Tensor>(allocator,prefill_shapes,act_dtype);
            prefill_attn_out_=std::make_shared<Tensor>(allocator,prefill_shapes,act_dtype);
            prefill_ffn_out_=std::make_shared<Tensor>(allocator,prefill_shapes,act_dtype);
            prefill_add_out_=std::make_shared<Tensor>(allocator,prefill_shapes,act_dtype);
            DeviceType dev=allocator->device_type();
            norm_out_->tensor_set_device_type(dev);
            attn_out_->tensor_set_device_type(dev);
            ffn_out_->tensor_set_device_type(dev);
            prefill_norm_out_->tensor_set_device_type(dev);
            prefill_attn_out_->tensor_set_device_type(dev);
            prefill_ffn_out_->tensor_set_device_type(dev);
            prefill_add_out_->tensor_set_device_type(dev);
            attn_norm_=std::make_shared<RMSNormLayer>(attn_norm_w);
            attentionLayer_=std::make_shared<AttentionLayer>(allocator,config_,wq,wk,wv,wo,bq,bk,bv);
            ffn_norm_=std::make_shared<RMSNormLayer>(ffn_norm_w);
            swigluLayer_=std::make_shared<SwigluLayer>(allocator,config_,w_gate,w_up,w_down);
            if(enable_fused) swigluLayer_->enable_fused();
        }

        TransformerLayer(std::shared_ptr<Allocator> allocator,ModelConfig& modelConfig,
                         std::shared_ptr<Tensor>& attn_norm_w,std::shared_ptr<Tensor>& wq,
                         std::shared_ptr<Tensor>& wk,std::shared_ptr<Tensor>& wv,std::shared_ptr<Tensor>& wo,
                         std::shared_ptr<Tensor>& ffn_norm_w,std::shared_ptr<Tensor>& w_gate,
                         std::shared_ptr<Tensor>& w_up,std::shared_ptr<Tensor>& w_down): Layer("Transformer"),
                         config_(modelConfig){
            int dim=config_.dim;
            std::vector<int> buffer_shapes={1,dim};
            norm_out_=std::make_shared<Tensor>(allocator,buffer_shapes,DataType::kDataTypeFP32);
            attn_out_=std::make_shared<Tensor>(allocator,buffer_shapes,DataType::kDataTypeFP32);
            ffn_out_=std::make_shared<Tensor>(allocator,buffer_shapes,DataType::kDataTypeFP32);
            DeviceType dev=allocator->device_type();
            norm_out_->tensor_set_device_type(dev);
            attn_out_->tensor_set_device_type(dev);
            ffn_out_->tensor_set_device_type(dev);
            attn_norm_=std::make_shared<RMSNormLayer>(attn_norm_w);
            attentionLayer_=std::make_shared<AttentionLayer>(allocator,config_,wq,wk,wv,wo);
            ffn_norm_=std::make_shared<RMSNormLayer>(ffn_norm_w);
            swigluLayer_=std::make_shared<SwigluLayer>(allocator,config_,w_gate,w_up,w_down);
        }

        std::shared_ptr<AttentionLayer>& get_attention() { return attentionLayer_; }

        void forward(std::shared_ptr<Tensor>& input, std::shared_ptr<Tensor>& output, int pos);
        void forward_prefill(std::shared_ptr<Tensor>& input, std::shared_ptr<Tensor>& output, int seq_len);

        void forward(std::shared_ptr<Tensor>& input, std::shared_ptr<Tensor>& output) override {
            throw std::runtime_error("TransformerLayer requires pos parameter!\n");
        }
    };
}
#endif
