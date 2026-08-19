#include <chrono>
#include "tensor/tensor.h"
#include "iostream"
#include "cuda_runtime.h"
#include "op/math_ops.h"
using namespace hxinfer;

bool check_accuracy(std::shared_ptr<Tensor>& cpu_tensor,std::shared_ptr<Tensor>& gpu_tensor){
    float max_diff=0.0f;
    size_t total_elements_cpu=cpu_tensor->tensor_total_elements();
    size_t total_elements_gpu=gpu_tensor->tensor_total_elements();
    float *cpu_ptr=cpu_tensor->tensor_data_ptr<float>();
    float *gpu_ptr=gpu_tensor->tensor_data_ptr<float>();
    if(total_elements_cpu!=total_elements_gpu){
        std::cerr<<"cpu/gpu输出维度都不一致\n";
    }else{
        for(size_t i=0;i<total_elements_gpu;i++){
            float diff=std::abs(gpu_ptr[i]-cpu_ptr[i]);
            if(diff>max_diff){
                max_diff=diff;
            }
        }
    }
    if(max_diff>1e-5){
        std::cerr<<"精度校验-FAIL"<<std::endl;
        return false;
    }else{
        std::cerr<<"精度校验-SUCCESS"<<std::endl;
        return true;
    }
}

__global__ void my_silu_kernel(const float *input,float *output,size_t total_elements){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<total_elements){
        float val=input[idx];
        output[idx]=val/(1+expf(-val));
    }
}

void my_silu_cuda(const float *input,float *output,size_t total_elements){
    int threads_per_block=256;
    int blocks_per_grid=(total_elements+threads_per_block-1)/threads_per_block;
    my_silu_kernel<<<blocks_per_grid,threads_per_block>>>(input,output,total_elements);
}
int main(){
    std::vector<int> shapes={1,4096,4096};
    std::shared_ptr<CPUAllocator> allocator_cpu=std::make_shared<CPUAllocator>();
    std::shared_ptr<CUDAAllocator> allocator_cuda=std::make_shared<CUDAAllocator>();
    std::shared_ptr<Tensor> h_in=std::make_shared<Tensor>(allocator_cpu,shapes,DataType::kDataTypeFP32);
    std::shared_ptr<Tensor> h_out=std::make_shared<Tensor>(allocator_cpu,shapes,DataType::kDataTypeFP32);
    std::shared_ptr<Tensor> c_out_my=std::make_shared<Tensor>(allocator_cpu,shapes,DataType::kDataTypeFP32);
    std::shared_ptr<Tensor> c_out_ai=std::make_shared<Tensor>(allocator_cpu,shapes,DataType::kDataTypeFP32);
    std::shared_ptr<Tensor> c_in_cuda_my=std::make_shared<Tensor>(allocator_cuda,shapes,DataType::kDataTypeFP32);
    std::shared_ptr<Tensor> c_out_cuda_my=std::make_shared<Tensor>(allocator_cuda,shapes,DataType::kDataTypeFP32);
    std::shared_ptr<Tensor> c_in_cuda_ai=std::make_shared<Tensor>(allocator_cuda,shapes,DataType::kDataTypeFP32);
    std::shared_ptr<Tensor> c_out_cuda_ai=std::make_shared<Tensor>(allocator_cuda,shapes,DataType::kDataTypeFP32);
    size_t total_elements=h_in->tensor_total_elements();
    float *h_in_ptr=h_in->tensor_data_ptr<float>();
    for(size_t i=0;i<total_elements;i++){
        h_in_ptr[i]=static_cast<float >(rand())/RAND_MAX*2.0-1.0;
    }
    std::cout << "========== SiLU Benchmark 开始 (数据量: " << total_elements << ") ==========\n";

    // ---------------------------------------------------------
    // [测试 1] CPU 基准测试
    // ---------------------------------------------------------
    auto cpu_start = std::chrono::high_resolution_clock::now();
    int num_runs = 50; // 测速循环次数
    for(int i=0; i<num_runs; i++) {
        silu_cpu(h_in,h_out);
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_avg_us = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count() / (float)num_runs;
    std::cout << "[1] CPU SiLU 平均耗时: " << cpu_avg_us << " us\n";
//    std::cout<<"初始化的cpu上的tensor_input"<<"\n";
//    h_in->tensor_print_data();
//    silu_cpu(h_in,h_out);
//    std::cout<<"cpu上的进行silu运算得到tensor_output"<<"\n";
//    h_out->tensor_print_data();


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    c_in_cuda_my=h_in->tensor_to_cuda(allocator_cuda);
    float *c_in_cuda_my_ptr=c_in_cuda_my->tensor_data_ptr<float>();
    float *c_out_cuda_my_ptr=c_out_cuda_my->tensor_data_ptr<float>();
    my_silu_cuda(c_in_cuda_my_ptr,c_out_cuda_my_ptr,total_elements);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for(int i=0; i<num_runs; i++) {
        my_silu_cuda(c_in_cuda_my_ptr, c_out_cuda_my_ptr, total_elements);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float my_ms = 0;
    cudaEventElapsedTime(&my_ms, start, stop); // 返回的是毫秒
    float my_avg_us = (my_ms * 1000.0f) / num_runs;
    std::cout << "[2] 手写 CUDA SiLU 平均耗时: " << my_avg_us << " us\n";
    float bandwith_VRAM_SM_my=(total_elements*4*2)/(my_avg_us/1e6)/(1024*1024*1024);
    std::cout << "[2] 手写 CUDA SiLU VRAM到SM之间的带宽: " << bandwith_VRAM_SM_my << " GB/S\n";


    c_out_my=c_out_cuda_my->tensor_to_cpu(allocator_cpu);

    std::cout<<"cuda上的进行silu运算得到tensor_output,自己写的"<<"\n";
    //c_out_my->tensor_print_data();
    if(check_accuracy(h_out,c_out_my)){
        std::cout<<"自己手写的silu与CPU版本的完全对齐\n";
    }else{
        std::cout<<"自己手写的silu出问题了\n";
    }


    c_in_cuda_ai=h_in->tensor_to_cuda(allocator_cuda);
    silu_cuda(c_in_cuda_ai,c_out_cuda_ai);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for(int i=0; i<num_runs; i++) {
        silu_cuda(c_in_cuda_ai, c_out_cuda_ai);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ai_ms = 0;
    cudaEventElapsedTime(&ai_ms, start, stop);
    float ai_avg_us = (ai_ms * 1000.0f) / num_runs;
    std::cout << "[3] AI 版 CUDA SiLU 平均耗时: " << ai_avg_us << " us\n";
    float bandwith_VRAM_SM_ai=(total_elements*4*2)/(ai_avg_us/1e6)/(1024*1024*1024);
    std::cout << "[3] AI CUDA SiLU VRAM到SM之间的带宽: " << bandwith_VRAM_SM_ai << " GB/S\n";
    c_out_ai=c_out_cuda_ai->tensor_to_cpu(allocator_cpu);
    std::cout<<"cuda上的进行silu运算得到tensor_output,ai写的"<<"\n";
    //c_out_ai->tensor_print_data();

    if(check_accuracy(h_out,c_out_ai)){
        std::cout<<"ai写的silu与CPU版本的完全对齐\n";
    }else{
        std::cout<<"ai写的silu出问题了\n";
    }
}
