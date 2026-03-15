#include "op/math_ops.h"
#include "iostream"
#include "tensor/tensor.h"
#include "memory"
#include "base/allocator.h"
#include "cuda_runtime.h"
using namespace hxinfer;
int main(){
    std::cout<<"-----开始进行silu的测试-----"<<std::endl;
    std::vector<int> shapes={2,3};
    std::shared_ptr<CPUAllocator> cpu_allocator=std::make_shared<CPUAllocator>();
    std::shared_ptr<CUDAAllocator> cuda_allocator=std::make_shared<CUDAAllocator>();
    std::cout<<"-----CPU-----"<<std::endl;
    std::shared_ptr<Tensor> input_cpu=std::make_shared<Tensor>(cpu_allocator,shapes,
                                                           DataType::kDataTypeFP32);
    std::shared_ptr<Tensor> output_cpu=std::make_shared<Tensor>(cpu_allocator,shapes,
                                                           DataType::kDataTypeFP32);
    float *in_cpu_ptr=input_cpu->tensor_data_ptr<float>();
    in_cpu_ptr[0]=1;in_cpu_ptr[1]=-1;in_cpu_ptr[2]=2;
    in_cpu_ptr[3]=-2;in_cpu_ptr[4]=3;in_cpu_ptr[5]=-3;
    silu_tensor(input_cpu,output_cpu);
    output_cpu->tensor_print_data();
    std::cout<<"-----CUDA-----"<<std::endl;
    std::shared_ptr<Tensor> input_cuda=input_cpu->to_cuda(cuda_allocator);
    std::shared_ptr<Tensor> output_cuda=std::make_shared<Tensor>(cuda_allocator,shapes,DataType::kDataTypeFP32);
    silu_tensor(input_cuda,output_cuda);
    std::shared_ptr<Tensor> output_print=output_cuda->to_cpu(cpu_allocator);
    cudaDeviceSynchronize();
    output_print->tensor_print_data();
}
