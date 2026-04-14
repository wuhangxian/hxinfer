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

        ModelConfig config_;

    public:
        TransformerLayer(std::shared_ptr<Allocator> allocator,ModelConfig& modelConfig,
                         std::shared_ptr<Tensor>& attn_norm_w,std::shared_ptr<Tensor>& wq,
                         std::shared_ptr<Tensor>& wk,std::shared_ptr<Tensor>& wv,std::shared_ptr<Tensor>& wo,
                         std::shared_ptr<Tensor>& ffn_norm_w,std::shared_ptr<Tensor>& w_gate,
                         std::shared_ptr<Tensor>& w_up,std::shared_ptr<Tensor>& w_down): Layer("Transformer"),
                                                                                         config_(modelConfig){
            int dim=config_.dim;
            int max_seq_len=config_.seq_len;
            DataType dtype=wq->tensor_data_type();
            std::vector<int> buffer_shapes={1,dim};

            norm_out_=std::make_shared<Tensor>(allocator,buffer_shapes,dtype);
            attn_out_=std::make_shared<Tensor>(allocator,buffer_shapes,dtype);
            ffn_out_=std::make_shared<Tensor>(allocator,buffer_shapes,dtype);

            attn_norm_=std::make_shared<RMSNormLayer>(attn_norm_w);
            attentionLayer_=std::make_shared<AttentionLayer>(allocator,config_,wq,wk,wv,wo);
            ffn_norm_=std::make_shared<RMSNormLayer>(ffn_norm_w);
            swigluLayer_=std::make_shared<SwigluLayer>(allocator,config_,w_gate,w_up,w_down);

        }

        // ===================== Prefix Cache 专用接口 =====================
        // 为什么要暴露 AttentionLayer？
        // PrefixCacheManager 需要穿透到每一层的 AttentionLayer，
        // 才能拿到该层的 k_cache_ 和 v_cache_ 进行快照/恢复。
        // 调用链: LlamaModel::get_blocks() → TransformerLayer::get_attention() → AttentionLayer::get_k/v_cache()
        std::shared_ptr<AttentionLayer>& get_attention() { return attentionLayer_; }

        // 🚀 1. 这是你真正用来干活的流水线！(带 pos)
        void forward(std::shared_ptr<Tensor>& input, std::shared_ptr<Tensor>& output, int pos);

        // 🚀 2. 这是糊弄编译器的“基础合同”，防止 TransformerLayer 变成抽象类报错！
        void forward(std::shared_ptr<Tensor>& input, std::shared_ptr<Tensor>& output) override {
            throw std::runtime_error("TransformerLayer requires 'pos' parameter! Please use the 3-parameter version.\n");
        }
    };
}

#endif //HXINFER_TRANSFORMER_H
