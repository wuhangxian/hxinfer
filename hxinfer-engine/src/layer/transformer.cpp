#include "layer/transformer.h"
#include "op/math_ops.h"
namespace hxinfer{
    void TransformerLayer::forward(std::shared_ptr<Tensor>& input, std::shared_ptr<Tensor>& output,int pos) {
        // SGLang-style: fuse residual add + RMSNorm in FP32
        // Step 1: input_layernorm (just RMSNorm, no residual add needed - input IS the residual)
        attn_norm_->forward(input, norm_out_);
        // Step 2: attention
        attentionLayer_->forward(norm_out_, attn_out_, pos);
        // Step 3: fused (input + attn_out) -> residual_ (FP32 add), ffn_norm(residual_) -> norm_out_
        fused_add_rmsnorm_tensor(attn_out_, input, norm_out_, residual_,
                                  ffn_norm_->get_weight());
        // Step 4: FFN
        swigluLayer_->forward(norm_out_, ffn_out_);
        // Step 5: output = residual_ + ffn_out (simple add)
        add_tensor(residual_, ffn_out_, output);
    }
}