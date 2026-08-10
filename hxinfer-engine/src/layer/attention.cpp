#include "layer/attention.h"
#include "op/math_ops.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "cmath"

namespace hxinfer{

    extern "C" void add_bias_broadcast_fp16(__half* data, const __half* bias, int total, int width);
    extern "C" void add_bias_broadcast_fp32(float* data, const float* bias, int total, int width);
    void attention_prefill_cuda(
        const std::shared_ptr<Tensor>& Q,
        const std::shared_ptr<Tensor>& K,
        const std::shared_ptr<Tensor>& V,
        std::shared_ptr<Tensor>& output,
        ModelConfig& config, int seq_len);
    void rope_prefill_cuda(std::shared_ptr<Tensor>& q, std::shared_ptr<Tensor>& k,
                       ModelConfig& config, int seq_len, int start_pos, float base);

    void AttentionLayer::forward(std::shared_ptr<Tensor>& input, std::shared_ptr<Tensor>& output, int pos) {
        int dim=config_.dim;
        int head=config_.head;
        int head_dim=dim/head;

        if(input->tensor_device_type()==DeviceType::kDeviceCUDA){
            int kv_dim = config_.kv_head * head_dim;
            curr_q_->tensor_reshape({1, dim});
            curr_k_->tensor_reshape({1, kv_dim});
            curr_v_->tensor_reshape({1, kv_dim});
            after_qktv_->tensor_reshape({1, dim});

            matmul_qkv_cuda(input, wq_, wk_, wv_, curr_q_, curr_k_, curr_v_);

            if(has_bias_){
                int q_size = (int)curr_q_->tensor_total_elements();
                int k_size = (int)curr_k_->tensor_total_elements();
                int v_size = (int)curr_v_->tensor_total_elements();
                if(curr_q_->tensor_data_type() == DataType::kDataTypeFP16){
                    add_bias_broadcast_fp16(curr_q_->tensor_data_ptr<__half>(), bq_->tensor_data_ptr<__half>(), q_size, dim);
                    add_bias_broadcast_fp16(curr_k_->tensor_data_ptr<__half>(), bk_->tensor_data_ptr<__half>(), k_size, kv_dim);
                    add_bias_broadcast_fp16(curr_v_->tensor_data_ptr<__half>(), bv_->tensor_data_ptr<__half>(), v_size, kv_dim);
                } else {
                    add_bias_broadcast_fp32(curr_q_->tensor_data_ptr<float>(), bq_->tensor_data_ptr<float>(), q_size, dim);
                    add_bias_broadcast_fp32(curr_k_->tensor_data_ptr<float>(), bk_->tensor_data_ptr<float>(), k_size, kv_dim);
                    add_bias_broadcast_fp32(curr_v_->tensor_data_ptr<float>(), bv_->tensor_data_ptr<float>(), v_size, kv_dim);
                }
            }

            rope_tensor(curr_q_, curr_k_, config_, pos, config_.rope_theta);
            attention_score_cuda(curr_q_, curr_k_, curr_v_, k_cache_, v_cache_, after_qktv_, config_, pos);
        } else {
            q_proj_->forward(input,curr_q_);
            k_proj_->forward(input,curr_k_);
            v_proj_->forward(input,curr_v_);
            rope_tensor(curr_q_,curr_k_,config_,pos, config_.rope_theta);

            float *ptr_k_cache=k_cache_->tensor_data_ptr<float>();
            float *ptr_v_cache=v_cache_->tensor_data_ptr<float>();
            float *ptr_curr_q=curr_q_->tensor_data_ptr<float>();
            float *ptr_curr_k=curr_k_->tensor_data_ptr<float>();
            float *ptr_curr_v=curr_v_->tensor_data_ptr<float>();
            float *ptr_curr_qktv=after_qktv_->tensor_data_ptr<float>();
            memcpy(ptr_k_cache+pos*dim,ptr_curr_k,dim*sizeof(float));
            memcpy(ptr_v_cache+pos*dim,ptr_curr_v,dim*sizeof(float));
            std::vector<float> scores(pos+1);
            float scale=1.0/std::sqrt(head_dim);
            for(int i=0;i<head;i++){
                float *curr_q=ptr_curr_q+i*head_dim;
                for(int j=0;j<=pos;j++){
                    float *curr_k=ptr_k_cache+j*dim+i*head_dim;
                    float sum=0;
                    for(int sc=0;sc<head_dim;sc++){
                        sum=sum+curr_q[sc]*curr_k[sc];
                    }
                    scores[j]=sum*scale;
                }
                float max_score=scores[0];
                for(int d=1;d<=pos;d++){
                    if(scores[d]>max_score){
                        max_score=scores[d];
                    }
                }
                float sum=0;
                for(int d=0;d<=pos;d++){
                    scores[d]=std::exp(scores[d]-max_score);
                    sum=sum+scores[d];
                }
                for(int d=0;d<=pos;d++){
                    scores[d]=scores[d]/sum;
                }
                float *curr_qkvt=ptr_curr_qktv+i*head_dim;
                for(int j=0;j<head_dim;j++){
                    float sum=0;
                    for(int k=0;k<=pos;k++){
                        sum=sum+scores[k]*ptr_v_cache[k*dim+i*head_dim+j];
                    }
                    curr_qkvt[j]=sum;
                }
            }
        }
        o_proj_->forward(after_qktv_,output);
    }

    void AttentionLayer::forward_prefill(std::shared_ptr<Tensor>& input, std::shared_ptr<Tensor>& output, int seq_len) {
        int dim = config_.dim;
        int head = config_.head;
        int head_dim = dim / head;
        int kv_dim = config_.kv_head * head_dim;

        // Reshape QKV buffers for prefill: [seq_len, dim/kv_dim]
        std::vector<int> q_shape = {seq_len, dim};
        std::vector<int> kv_shape = {seq_len, kv_dim};
        curr_q_->tensor_reshape(q_shape);
        curr_k_->tensor_reshape(kv_shape);
        curr_v_->tensor_reshape(kv_shape);
        after_qktv_->tensor_reshape(q_shape);

        // QKV projection (cuBLAS handles M=seq_len automatically)
        matmul_qkv_cuda(input, wq_, wk_, wv_, curr_q_, curr_k_, curr_v_);

        // Add bias if present (broadcast: bias [dim] -> [seq_len, dim])
        if(has_bias_){
            int q_size = (int)curr_q_->tensor_total_elements();
            int k_size = (int)curr_k_->tensor_total_elements();
            int v_size = (int)curr_v_->tensor_total_elements();
            if(curr_q_->tensor_data_type() == DataType::kDataTypeFP16){
                add_bias_broadcast_fp16(curr_q_->tensor_data_ptr<__half>(), bq_->tensor_data_ptr<__half>(), q_size, dim);
                add_bias_broadcast_fp16(curr_k_->tensor_data_ptr<__half>(), bk_->tensor_data_ptr<__half>(), k_size, kv_dim);
                add_bias_broadcast_fp16(curr_v_->tensor_data_ptr<__half>(), bv_->tensor_data_ptr<__half>(), v_size, kv_dim);
            } else {
                add_bias_broadcast_fp32(curr_q_->tensor_data_ptr<float>(), bq_->tensor_data_ptr<float>(), q_size, dim);
                add_bias_broadcast_fp32(curr_k_->tensor_data_ptr<float>(), bk_->tensor_data_ptr<float>(), k_size, kv_dim);
                add_bias_broadcast_fp32(curr_v_->tensor_data_ptr<float>(), bv_->tensor_data_ptr<float>(), v_size, kv_dim);
            }
        }

        // RoPE prefill (each token at position 0..seq_len-1)
        rope_prefill_cuda(curr_q_, curr_k_, config_, seq_len, 0, config_.rope_theta);
        cudaDeviceSynchronize();

        // Prefill attention (causal mask, no KV cache)
        attention_prefill_cuda(curr_q_, curr_k_, curr_v_, after_qktv_, config_, seq_len);

        // Write K/V to KV cache for decode phase
        // k_cache: [max_seq_len, kv_dim], K: [seq_len, kv_dim]
        // Both are contiguous, so simple memcpy works
        __half* k_cache_ptr = k_cache_->tensor_data_ptr<__half>();
        __half* v_cache_ptr = v_cache_->tensor_data_ptr<__half>();
        const __half* k_ptr = curr_k_->tensor_data_ptr<__half>();
        const __half* v_ptr = curr_v_->tensor_data_ptr<__half>();
        cudaMemcpy(k_cache_ptr, k_ptr, (size_t)seq_len * (size_t)kv_dim * sizeof(__half), cudaMemcpyDeviceToDevice);
        cudaMemcpy(v_cache_ptr, v_ptr, (size_t)seq_len * (size_t)kv_dim * sizeof(__half), cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();

        // O projection
        o_proj_->forward(after_qktv_, output);
    }
}
