#ifndef HXINFER_EMBEDDING_H
#define HXINFER_EMBEDDING_H
#include "layer/layer.h"
#include "op/math_ops.h"
namespace hxinfer{
    class EmbeddingLayer:public Layer{
    private:
        std::shared_ptr<Tensor> weight_;
    public:
        EmbeddingLayer(std::shared_ptr<Tensor> weight): Layer("Embedding"),weight_(weight){}
        void forward(std::shared_ptr<Tensor>& input,std::shared_ptr<Tensor>& output) override{
            embedding_tensor(input,weight_,output);
        }
    };
}
#endif //HXINFER_EMBEDDING_H
