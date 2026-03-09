
#ifndef HXINFER_LINEAR_H
#define HXINFER_LINEAR_H

#include "layer.h"
#include "op/math_ops.h"
namespace hxinfer{
    class LinearLayer:public Layer{
    private:
        std::shared_ptr<Tensor> weight_;
    public:
        LinearLayer(std::shared_ptr<Tensor> weight): Layer("Linear"),
                                                     weight_(weight){}
        void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override{
            matmul_tensor(input,weight_,output);
        }
    };
}

#endif //HXINFER_LINEAR_H
