#include "layer/transformer.h"
#include "op/math_ops.h"
namespace hxinfer{
    void TransformerLayer::forward(std::shared_ptr<Tensor>& input, std::shared_ptr<Tensor>& output,int pos) {
        attn_norm_->forward(input,norm_out_);
        attentionLayer_->forward(norm_out_,attn_out_,pos);
        add_tensor(input,attn_out_,attn_out_);
        ffn_norm_->forward(attn_out_,norm_out_);
        swigluLayer_->forward(norm_out_,ffn_out_);
        add_tensor(attn_out_,ffn_out_,output);
    }
}