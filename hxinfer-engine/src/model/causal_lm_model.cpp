#include "model/causal_lm_model.h"
#include "op/math_ops.h"
namespace hxinfer{
    void CausalLMModel::forward(std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> &output,int pos) {
        embeddingLayer_->forward(input,ping_);
        for(int i=0;i<blocks_.size();i++){
            blocks_[i]->forward(ping_,pang_,pos);
            std::swap(ping_,pang_);
        }
        final_norm_->forward(ping_,ping_);
        lm_head_->forward(ping_,output);
    }

    void CausalLMModel::forward_prefill(std::shared_ptr<Tensor> &input_ids, std::shared_ptr<Tensor> &output, int seq_len) {
        // Reshape prefill buffers to [seq_len, dim]
        std::vector<int> prefill_shape = {seq_len, config_.dim};
        prefill_ping_->tensor_reshape(prefill_shape);
        prefill_pang_->tensor_reshape(prefill_shape);

        // Embedding: [seq_len] token ids -> [seq_len, dim] hidden
        embeddingLayer_->forward(input_ids, prefill_ping_);

        // Transformer layers (prefill mode)
        for(int i=0; i<blocks_.size(); i++){
            blocks_[i]->forward_prefill(prefill_ping_, prefill_pang_, seq_len);
            std::swap(prefill_ping_, prefill_pang_);
        }

        // Final norm + LM head
        final_norm_->forward(prefill_ping_, prefill_ping_);
        lm_head_->forward(prefill_ping_, output);
    }
}
