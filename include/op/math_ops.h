#ifndef HXINFER_MATH_OPS_H
#define HXINFER_MATH_OPS_H
#include "memory"
#include "tensor/tensor.h"

namespace hxinfer{
    void add_tensor(const std::shared_ptr<Tensor>& input_a,const std::shared_ptr<Tensor>& input_b,
                    std::shared_ptr<Tensor>& output);
    void mul_tensor(const std::shared_ptr<Tensor>& input_a,const std::shared_ptr<Tensor>& input_b,
                    std::shared_ptr<Tensor>& output);
    void softmax_tensor(const std::shared_ptr<Tensor>& input,std::shared_ptr<Tensor>& output);
    void matmul_tensor(const std::shared_ptr<Tensor>& input,const std::shared_ptr<Tensor>& weight,
                       std::shared_ptr<Tensor>& output);
    void rope_tensor(std::shared_ptr<Tensor>& q,std::shared_ptr<Tensor>& k,int step,int base=10000);
    void rmsnorm_tensor(const std::shared_ptr<Tensor>& input,const std::shared_ptr<Tensor>& weight,
                        std::shared_ptr<Tensor>& output,float eps=1e-5);
    void embedding_tensor(const std::shared_ptr<Tensor>& token_ids,const std::shared_ptr<Tensor>& weight,
                          std::shared_ptr<Tensor>& output);
    void silu_tensor(const std::shared_ptr<Tensor>& input,std::shared_ptr<Tensor>& output);
    int  argmax_tensor(const std::shared_ptr<Tensor>& input);
}

#endif //HXINFER_MATH_OPS_H
