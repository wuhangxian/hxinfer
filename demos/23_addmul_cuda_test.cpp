#include "iostream"
#include "op/math_ops.h"
#include "base/allocator.h"
using namespace hxinfer;
int main(){
    std::cout<<"-----add_tensor和mul_tensor-----"<<std::endl;
    std::vector<int> shapes={2,3};
    std::shared_ptr<CPUAllocator> allocator_cpu=std::make_shared<CPUAllocator>();
    std::shared_ptr<CUDAAllocator> allocator_cuda=std::make_shared<CUDAAllocator>();
    std::shared_ptr<Tensor> input_a_cpu=std::make_shared<Tensor>(allocator_cpu,shapes, DataType::kDataTypeFP32);
    std::shared_ptr<Tensor> input_b_cpu=std::make_shared<Tensor>(allocator_cpu,shapes, DataType::kDataTypeFP32);
    std::shared_ptr<Tensor> output_cpu=std::make_shared<Tensor>(allocator_cpu,shapes, DataType::kDataTypeFP32);
    float *in_a_cpu=input_a_cpu->tensor_data_ptr<float>();
    float *in_b_cpu=input_b_cpu->tensor_data_ptr<float>();
    for(int i=0;i<6;i++){
        in_a_cpu[i]=i;
        in_b_cpu[i]=i;
    }
    add_tensor(input_a_cpu,input_b_cpu,output_cpu);
    std::cout<<"-----add_tensor---CPU-----"<<std::endl;
    output_cpu->tensor_print_data();
    mul_tensor(input_a_cpu,input_b_cpu,output_cpu);
    std::cout<<"-----mul_tensor---CPU-----"<<std::endl;
    output_cpu->tensor_print_data();
    std::shared_ptr<Tensor> input_a_cuda=input_a_cpu->tensor_to_cuda(allocator_cuda);
    std::shared_ptr<Tensor> input_b_cuda=input_b_cpu->tensor_to_cuda(allocator_cuda);
    std::shared_ptr<Tensor> output_cuda=std::make_shared<Tensor>(allocator_cuda,shapes,DataType::kDataTypeFP32);
    std::shared_ptr<Tensor> output_print=std::make_shared<Tensor>(allocator_cpu,shapes,DataType::kDataTypeFP32);
    add_tensor(input_a_cuda,input_b_cuda,output_cuda);
    output_print=output_cuda->tensor_to_cpu(allocator_cpu);
    std::cout<<"-----add_tensor---CUDA-----"<<std::endl;
    output_print->tensor_print_data();
    mul_tensor(input_a_cuda,input_b_cuda,output_cuda);
    output_print=output_cuda->tensor_to_cpu(allocator_cpu);
    std::cout<<"-----mul_tensor---CUDA-----"<<std::endl;
    output_print->tensor_print_data();
}