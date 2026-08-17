#include "layer/attention.h"
#include "op/math_ops.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "cmath"

namespace hxinfer{

    // CUDA kernels declared as extern - implemented in a .cu file
    extern "C" void add_bias_cuda_fp16(__half* data, const __half* bias, int n);
    extern "C" void add_bias_cuda_fp32(float* data, const float* bias, int n);

    void AttentionLayer::forward(std::shared_ptr<Tensor>& input, std::shared_ptr<Tensor>& output, int pos) {
        int dim=config_.dim;
        int head=config_.head;
        int head_dim=dim/head;

        if(input->tensor_device_type()==DeviceType::kDeviceCUDA){
            matmul_qkv_cuda(input, wq_, wk_, wv_, curr_q_, curr_k_, curr_v_);

           if(has_bias_){
                int q_size = (int)curr_q_->tensor_total_elements();
                int kv_size = (int)curr_k_->tensor_total_elements();
               if(curr_q_->tensor_data_type() == DataType::kDataTypeFP16){
                    add_bias_cuda_fp16(curr_q_->tensor_data_ptr<__half>(), bq_->tensor_data_ptr<__half>(), q_size);
                    add_bias_cuda_fp16(curr_k_->tensor_data_ptr<__half>(), bk_->tensor_data_ptr<__half>(), kv_size);
                    add_bias_cuda_fp16(curr_v_->tensor_data_ptr<__half>(), bv_->tensor_data_ptr<__half>(), kv_size);
               } else {
                    add_bias_cuda_fp32(curr_q_->tensor_data_ptr<float>(), bq_->tensor_data_ptr<float>(), q_size);
                    add_bias_cuda_fp32(curr_k_->tensor_data_ptr<float>(), bk_->tensor_data_ptr<float>(), kv_size);
                    add_bias_cuda_fp32(curr_v_->tensor_data_ptr<float>(), bv_->tensor_data_ptr<float>(), kv_size);
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
}
