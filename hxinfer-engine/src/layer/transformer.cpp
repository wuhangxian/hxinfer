#include "layer/transformer.h"
#include "op/math_ops.h"
namespace hxinfer{
    void TransformerLayer::forward(std::shared_ptr<Tensor>& input, std::shared_ptr<Tensor>& output, int pos) {
        attn_norm_->forward(input, norm_out_);
        attentionLayer_->forward(norm_out_, attn_out_, pos);
        add_tensor(input, attn_out_, attn_out_);
        ffn_norm_->forward(attn_out_, norm_out_);
        swigluLayer_->forward(norm_out_, ffn_out_);
        add_tensor(attn_out_, ffn_out_, output);
    }

    void TransformerLayer::forward_prefill(std::shared_ptr<Tensor>& input, std::shared_ptr<Tensor>& output, int seq_len) {
        // Reshape prefill buffers
        std::vector<int> prefill_shape = {seq_len, config_.dim};
        prefill_norm_out_->tensor_reshape(prefill_shape);
        prefill_attn_out_->tensor_reshape(prefill_shape);
        prefill_ffn_out_->tensor_reshape(prefill_shape);
        prefill_add_out_->tensor_reshape(prefill_shape);

        attn_norm_->forward(input, prefill_norm_out_);
        attentionLayer_->forward_prefill(prefill_norm_out_, prefill_attn_out_, seq_len);
        add_tensor(input, prefill_attn_out_, prefill_attn_out_);
        ffn_norm_->forward(prefill_attn_out_, prefill_norm_out_);
        swigluLayer_->forward(prefill_norm_out_, prefill_ffn_out_);
        add_tensor(prefill_attn_out_, prefill_ffn_out_, output);
    }
}
