#include "model/llama_model.h"
#include "op/math_ops.h"
namespace hxinfer{
    void LlamaModel::forward(std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> &output,int pos) {
        embeddingLayer_->forward(input,ping_);
        for(int i=0;i<blocks_.size();i++){
            blocks_[i]->forward(ping_,pang_,pos);
            std::swap(ping_,pang_);
        }
        final_norm_->forward(ping_,ping_);
        lm_head_->forward(ping_,output);
    }
}

