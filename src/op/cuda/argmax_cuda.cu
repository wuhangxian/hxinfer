#include "tensor/tensor.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "iostream"
namespace hxinfer{
    // FP32 argmax kernel
    __global__ void argmax_kernel_cuda(const float* data,int* result,size_t total_elements){
        extern __shared__ char shared_mem[];
        float* shared_vals=reinterpret_cast<float*>(shared_mem);
        int* shared_idxs=reinterpret_cast<int*>(shared_mem+blockDim.x*sizeof(float));

        int tid=threadIdx.x;
        int block_size=blockDim.x;

        float local_max=-3.402823466e+38f;
        int local_idx=0;
        for(size_t i=tid;i<total_elements;i+=block_size){
            if(data[i]>local_max){
                local_max=data[i];
                local_idx=static_cast<int>(i);
            }
        }
        shared_vals[tid]=local_max;
        shared_idxs[tid]=local_idx;
        __syncthreads();

        for(int stride=block_size/2;stride>0;stride>>=1){
            if(tid<stride){
                if(shared_vals[tid+stride]>shared_vals[tid]){
                    shared_vals[tid]=shared_vals[tid+stride];
                    shared_idxs[tid]=shared_idxs[tid+stride];
                }
            }
            __syncthreads();
        }

        if(tid==0){
            *result=shared_idxs[0];
        }
    }

    // FP16 argmax kernel
    __global__ void argmax_kernel_cuda_fp16(const __half* data,int* result,size_t total_elements){
        extern __shared__ char shared_mem[];
        float* shared_vals=reinterpret_cast<float*>(shared_mem);
        int* shared_idxs=reinterpret_cast<int*>(shared_mem+blockDim.x*sizeof(float));

        int tid=threadIdx.x;
        int block_size=blockDim.x;

        float local_max=-3.402823466e+38f;
        int local_idx=0;
        for(size_t i=tid;i<total_elements;i+=block_size){
            float val=__half2float(data[i]);
            if(val>local_max){
                local_max=val;
                local_idx=static_cast<int>(i);
            }
        }
        shared_vals[tid]=local_max;
        shared_idxs[tid]=local_idx;
        __syncthreads();

        for(int stride=block_size/2;stride>0;stride>>=1){
            if(tid<stride){
                if(shared_vals[tid+stride]>shared_vals[tid]){
                    shared_vals[tid]=shared_vals[tid+stride];
                    shared_idxs[tid]=shared_idxs[tid+stride];
                }
            }
            __syncthreads();
        }

        if(tid==0){
            *result=shared_idxs[0];
        }
    }

    int argmax_cuda(const std::shared_ptr<Tensor>& input){
        if(input->tensor_device_type()!=DeviceType::kDeviceCUDA){
            std::cerr<<"[Fatal Error] argmax_cuda expects CUDA Tensor!"<<std::endl;
            return -1;
        }
        size_t total_elements=input->tensor_total_elements();
        if(total_elements==0) return -1;

        static int* d_result=[](){
            int* p;
            cudaMalloc(&p,sizeof(int));
            return p;
        }();

        int threads=256;

        if(input->tensor_data_type()==DataType::kDataTypeFP16){
            const __half* d_in=input->tensor_data_ptr<__half>();
            size_t shared_bytes=threads*(sizeof(float)+sizeof(int));
            argmax_kernel_cuda_fp16<<<1,threads,shared_bytes>>>(d_in,d_result,total_elements);
        } else {
            const float* d_in=input->tensor_data_ptr<float>();
            size_t shared_bytes=threads*(sizeof(float)+sizeof(int));
            argmax_kernel_cuda<<<1,threads,shared_bytes>>>(d_in,d_result,total_elements);
        }

        int h_result=0;
        cudaMemcpy(&h_result,d_result,sizeof(int),cudaMemcpyDeviceToHost);
        return h_result;
    }
}
