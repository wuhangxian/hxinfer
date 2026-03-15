#include "tensor/tensor.h"
#include "iostream"
#include "cublas_v2.h"
#include "cuda_runtime.h"
namespace hxinfer{
    void matmul_cuda(const std::shared_ptr<Tensor>& input,const std::shared_ptr<Tensor>& weight,
                     std::shared_ptr<Tensor>& output){
        if(input->tensor_device_type()!=DeviceType::kDeviceCUDA||
                weight->tensor_device_type()!=DeviceType::kDeviceCUDA||
                output->tensor_device_type()!=DeviceType::kDeviceCUDA){
            std::cerr<<"[Fatal Error] matmul_cuda expects all Tensors to be on CUDA"<<std::endl;
            return;
        }
        cublasHandle_t handle;
        if(cublasCreate(&handle)!=CUBLAS_STATUS_SUCCESS){
            std::cerr<<"CUBLAS initialization failed!"<<std::endl;
            return;
        }
        const float* d_A=input->tensor_data_ptr<float>();// 假设形状 [M, K]
        const float* d_B=weight->tensor_data_ptr<float>();// 假设形状 [N, K]
        float* d_C=output->tensor_data_ptr<float>();// 输出形状 [M, N]

        int K=weight->tensor_shapes()[1];
        int M=input->tensor_total_elements()/K;
        int N=output->tensor_total_elements()/M;

        float alpha=1.0;
        float beta=0;

        cublasStatus_t stat=cublasSgemm(
                handle,
                CUBLAS_OP_T,CUBLAS_OP_N,
                N,M,K,
                &alpha,
                d_B,K,
                d_A,K,
                &beta,
                d_C,N
                );

        if(stat!=CUBLAS_STATUS_SUCCESS){
            std::cerr<<"CUBLAS SGEMM failed!"<<std::endl;
        }
        cublasDestroy(handle);
        cudaDeviceSynchronize();
    }

}
