#include "tensor/tensor.h"
#include "cuda_runtime.h"
#include "iostream"
namespace hxinfer{

    // KV Cache 更新 Kernel：把当前步的 K、V 写入 Cache 的第 pos 行
    __global__ void kv_cache_update_kernel(float* k_cache,float* v_cache,
                                           const float* curr_k,const float* curr_v,
                                           int dim,int pos){
        int idx=blockIdx.x*blockDim.x+threadIdx.x;
        if(idx<dim){
            k_cache[pos*dim+idx]=curr_k[idx];
            v_cache[pos*dim+idx]=curr_v[idx];
        }
    }

    // 注意力得分计算 Kernel：每个 Block 处理一个 Head
    // 流程: QK^T → softmax → score*V
    __global__ void attention_score_kernel(const float* q,const float* k_cache,const float* v_cache,
                                           float* output,int head_dim,int dim,int pos,float scale){
        int head_idx=blockIdx.x;
        int tid=threadIdx.x;
        int block_size=blockDim.x;

        const float* q_head=q+head_idx*head_dim;
        int seq_len=pos+1;

        // 动态共享内存布局: [scores: seq_len floats] [temp: block_size floats]
        extern __shared__ float shared[];
        float* scores=shared;
        float* temp=shared+seq_len;

        // ========== Step 1: 计算 Q · K^T 得分 ==========
        for(int p=tid;p<=pos;p+=block_size){
            const float* k_head=k_cache+p*dim+head_idx*head_dim;
            float dot=0.0f;
            for(int d=0;d<head_dim;d++){
                dot+=q_head[d]*k_head[d];
            }
            scores[p]=dot*scale;
        }
        __syncthreads();

        // ========== Step 2: Softmax on scores ==========
        // 2a: 找最大值
        float local_max=-3.402823466e+38f;
        for(int p=tid;p<=pos;p+=block_size){
            if(scores[p]>local_max) local_max=scores[p];
        }
        temp[tid]=local_max;
        __syncthreads();
        for(int s=block_size/2;s>0;s>>=1){
            if(tid<s && temp[tid+s]>temp[tid]){
                temp[tid]=temp[tid+s];
            }
            __syncthreads();
        }
        float max_val=temp[0];
        __syncthreads();

        // 2b: exp 并求和
        float local_sum=0.0f;
        for(int p=tid;p<=pos;p+=block_size){
            scores[p]=expf(scores[p]-max_val);
            local_sum+=scores[p];
        }
        temp[tid]=local_sum;
        __syncthreads();
        for(int s=block_size/2;s>0;s>>=1){
            if(tid<s) temp[tid]+=temp[tid+s];
            __syncthreads();
        }
        float sum_val=temp[0];
        __syncthreads();

        // 2c: 归一化
        for(int p=tid;p<=pos;p+=block_size){
            scores[p]/=sum_val;
        }
        __syncthreads();

        // ========== Step 3: score * V 加权求和 ==========
        for(int d=tid;d<head_dim;d+=block_size){
            float sum=0.0f;
            for(int p=0;p<=pos;p++){
                sum+=scores[p]*v_cache[p*dim+head_idx*head_dim+d];
            }
            output[head_idx*head_dim+d]=sum;
        }
    }

    void attention_score_cuda(const std::shared_ptr<Tensor>& q,
                              const std::shared_ptr<Tensor>& curr_k,
                              const std::shared_ptr<Tensor>& curr_v,
                              std::shared_ptr<Tensor>& k_cache,
                              std::shared_ptr<Tensor>& v_cache,
                              std::shared_ptr<Tensor>& output,
                              ModelConfig& config,int pos){
        int dim=config.dim;
        int head=config.head;
        int head_dim=dim/head;
        float scale=1.0f/sqrtf(static_cast<float>(head_dim));

        const float* d_q=q->tensor_data_ptr<float>();
        const float* d_curr_k=curr_k->tensor_data_ptr<float>();
        const float* d_curr_v=curr_v->tensor_data_ptr<float>();
        float* d_k_cache=k_cache->tensor_data_ptr<float>();
        float* d_v_cache=v_cache->tensor_data_ptr<float>();
        float* d_out=output->tensor_data_ptr<float>();

        // 1) 更新 KV Cache
        int threads_update=256;
        int blocks_update=(dim+threads_update-1)/threads_update;
        kv_cache_update_kernel<<<blocks_update,threads_update>>>(d_k_cache,d_v_cache,
                                                                  d_curr_k,d_curr_v,dim,pos);

        // 2) 注意力得分计算：每个 Head 一个 Block
        int threads_attn=128;
        int seq_len=pos+1;
        size_t shared_bytes=(seq_len+threads_attn)*sizeof(float);
        attention_score_kernel<<<head,threads_attn,shared_bytes>>>(d_q,d_k_cache,d_v_cache,
                                                                    d_out,head_dim,dim,pos,scale);
        cudaDeviceSynchronize();
    }
}
