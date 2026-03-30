#include "tensor/tensor.h"
//#include "op/math_ops.h"
#include "iostream"
#include "cuda_runtime.h"
namespace hxinfer{
    __global__ void rmsnorm_kernel_cuda(float *out_data,const float* in_data,
                                        const float *weight,int hidden_dim,float eps){
        int tid=threadIdx.x;
        int bid=blockIdx.x;

        const float *current_in=in_data+bid*hidden_dim;
        float *current_out=out_data+bid*hidden_dim;

        float local_sum=0;
        for(int i=tid;i<hidden_dim;i=i+blockDim.x){
            local_sum=local_sum+current_in[i]+current_in[i];
        }
        extern __shared__ float s_sum[];
        s_sum[tid]=local_sum;
        __syncthreads();

        for(int stride=blockDim.x/2;stride>0;stride>>1){
            if(tid<stride){
                s_sum[tid]=s_sum[tid]+s_sum[tid+stride];
            }
            __syncthreads();
        }
        __shared__ float s_inv_rms;
        if(tid==0){
            s_inv_rms= rsqrtf(s_sum[0]/hidden_dim+eps);
        }
        __syncthreads();
        for(int i=tid;i<hidden_dim;i += blockDim.x){
            current_out[i]=current_in[i]*s_inv_rms*weight[i];
        }

    }

    void rmsnorm_cuda(const std::shared_ptr<Tensor>& input,const std::shared_ptr<Tensor>& weight,
                      std::shared_ptr<Tensor>& output,float eps=1e-5){
        // 【规矩 1】：严格的设备类型校验
        if(input->tensor_device_type() != DeviceType::kDeviceCUDA ||
           weight->tensor_device_type() != DeviceType::kDeviceCUDA ||
           output->tensor_device_type() != DeviceType::kDeviceCUDA){
            std::cerr << "[Fatal Error] rmsnorm_cuda expects CUDA Tensors!" << std::endl;
            return;
        }
        size_t  total_elements=input->tensor_total_elements();
        size_t  hidden_dim=weight->tensor_total_elements();
        // 【规矩 2】：空数据拦截
        if(total_elements==0||hidden_dim==0){
            return;
        }
        size_t num_tokens=total_elements/hidden_dim;

        const float *d_in=input->tensor_data_ptr<float>();
        const float *d_weight=weight->tensor_data_ptr<float>();
        float  *d_out=output->tensor_data_ptr<float>();

// 【规矩 4】：极其关键的 Grid 和 Block 划分 (与 SiLU 截然不同)
        int threads_per_block = 256;         // 每个 Token 用 256 个线程并发处理
        int blocks_per_grid = num_tokens;    // 有多少个 Token，就派多少个 Block

        size_t shared_mem_bytes=threads_per_block*sizeof (float );

        rmsnorm_kernel_cuda<<<blocks_per_grid,threads_per_block,shared_mem_bytes>>>(d_out,
                                    d_int,d_weight,hidden_dim,eps);
        cudaDeviceSynchronize();

    }
}