#ifndef HXINFER_MATH_OPS_H
#define HXINFER_MATH_OPS_H
#include "memory"
#include "tensor/tensor.h"

namespace hxinfer{
    void add_tensor(const std::shared_ptr<Tensor>& input_a,const std::shared_ptr<Tensor>& input_b,
                    std::shared_ptr<Tensor>& output);
    void add_cpu(const std::shared_ptr<Tensor>& input_a,const std::shared_ptr<Tensor>& input_b,
                    std::shared_ptr<Tensor>& output);
    void add_cuda(const std::shared_ptr<Tensor>& input_a,const std::shared_ptr<Tensor>& input_b,
                    std::shared_ptr<Tensor>& output);
    void mul_tensor(const std::shared_ptr<Tensor>& input_a,const std::shared_ptr<Tensor>& input_b,
                    std::shared_ptr<Tensor>& output);
    void mul_cpu(const std::shared_ptr<Tensor>& input_a,const std::shared_ptr<Tensor>& input_b,
                    std::shared_ptr<Tensor>& output);
    void mul_cuda(const std::shared_ptr<Tensor>& input_a,const std::shared_ptr<Tensor>& input_b,
                    std::shared_ptr<Tensor>& output);
    void softmax_tensor(const std::shared_ptr<Tensor>& input,std::shared_ptr<Tensor>& output);
    void softmax_cpu(const std::shared_ptr<Tensor>& input,std::shared_ptr<Tensor>& output);
    void softmax_cuda(const std::shared_ptr<Tensor>& input,std::shared_ptr<Tensor>& output);
    void matmul_tensor(const std::shared_ptr<Tensor>& input,const std::shared_ptr<Tensor>& weight,
                       std::shared_ptr<Tensor>& output);
    void matmul_cpu(const std::shared_ptr<Tensor>& input,const std::shared_ptr<Tensor>& weight,
                       std::shared_ptr<Tensor>& output);
    void matmul_cuda(const std::shared_ptr<Tensor>& input,const std::shared_ptr<Tensor>& weight,
                       std::shared_ptr<Tensor>& output);
    void rope_tensor(std::shared_ptr<Tensor>& q,std::shared_ptr<Tensor>& k,ModelConfig& config,int step,float base=10000);
    void rope_cpu(std::shared_ptr<Tensor>& q,std::shared_ptr<Tensor>& k,ModelConfig& config,int step,float base=10000);
    void rope_cuda(std::shared_ptr<Tensor>& q,std::shared_ptr<Tensor>& k,ModelConfig& config,int step,float base=10000);
    void rmsnorm_tensor(const std::shared_ptr<Tensor>& input,const std::shared_ptr<Tensor>& weight,
                        std::shared_ptr<Tensor>& output,float eps=1e-5);
    void rmsnorm_cpu(const std::shared_ptr<Tensor>& input,const std::shared_ptr<Tensor>& weight,
                        std::shared_ptr<Tensor>& output,float eps=1e-5);
    void rmsnorm_cuda(const std::shared_ptr<Tensor>& input,const std::shared_ptr<Tensor>& weight,
                        std::shared_ptr<Tensor>& output,float eps=1e-5);
    void embedding_tensor(const std::shared_ptr<Tensor>& token_ids,const std::shared_ptr<Tensor>& weight,
                          std::shared_ptr<Tensor>& output);
    void embedding_cpu(const std::shared_ptr<Tensor>& token_ids,const std::shared_ptr<Tensor>& weight,
                       std::shared_ptr<Tensor>& output);
    void embedding_cuda(const std::shared_ptr<Tensor>& token_ids,const std::shared_ptr<Tensor>& weight,
                        std::shared_ptr<Tensor>& output);
    void silu_tensor(const std::shared_ptr<Tensor>& input,std::shared_ptr<Tensor>& output);
    void silu_cpu(const std::shared_ptr<Tensor>& input, std::shared_ptr<Tensor>& output);
    void silu_cuda(const std::shared_ptr<Tensor>& input, std::shared_ptr<Tensor>& output);
    int  argmax_tensor(const std::shared_ptr<Tensor>& input);
    int  argmax_cpu(const std::shared_ptr<Tensor>& input);
    int  argmax_cuda(const std::shared_ptr<Tensor>& input);

    // Temperature + Top-p 采样
    int  sample_tensor(const std::shared_ptr<Tensor>& logits, float temperature, float top_p);

    // 融合 QKV / Gate+Up 投影
    void matmul_qkv_cuda(const std::shared_ptr<Tensor>& input,
                          const std::shared_ptr<Tensor>& wq,
                          const std::shared_ptr<Tensor>& wk,
                          const std::shared_ptr<Tensor>& wv,
                          std::shared_ptr<Tensor>& q_out,
                          std::shared_ptr<Tensor>& k_out,
                          std::shared_ptr<Tensor>& v_out);
    void matmul_gate_up_cuda(const std::shared_ptr<Tensor>& input,
                              const std::shared_ptr<Tensor>& w_gate,
                              const std::shared_ptr<Tensor>& w_up,
                              std::shared_ptr<Tensor>& gate_out,
                              std::shared_ptr<Tensor>& up_out);

    void attention_score_cuda(const std::shared_ptr<Tensor>& q,
                              const std::shared_ptr<Tensor>& curr_k,
                              const std::shared_ptr<Tensor>& curr_v,
                              std::shared_ptr<Tensor>& k_cache,
                              std::shared_ptr<Tensor>& v_cache,
                              std::shared_ptr<Tensor>& output,
                              ModelConfig& config,int pos);
}

#endif //HXINFER_MATH_OPS_H
