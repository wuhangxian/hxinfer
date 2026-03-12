#ifndef HXINFER_LAYER_H
#define HXINFER_LAYER_H
#include "string"
#include "tensor/tensor.h"
#include "memory"
namespace hxinfer{
    class Layer{
    public:
        std::string layer_name_;
        Layer(std::string layer_name):layer_name_(layer_name){}
        virtual void forward(std::shared_ptr<Tensor>& input,std::shared_ptr<Tensor>& output)=0;
        virtual ~Layer()=default;
    };
}

#endif //HXINFER_LAYER_H
