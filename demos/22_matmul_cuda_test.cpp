#include "iostream"

#include "op/math_ops.h"
using namespace hxinfer;
int main(){
    std::shared_ptr<CPUAllocator> allocator_cpu=std::make_shared<CPUAllocator>();
    std::shared_ptr<CUDAAllocator> allocator_cuda=std::make_shared<CUDAAllocator>();
    std::cout<<"-----开始进行matmul的测试-----"<<std::endl;
    std::vector<int> in_shapes={2,3};
    std::vector<int> w_shapes={4,3};
    std::vector<int> out_shapes={2,4};
    std::cout<<"-----CPU-----\n";
    std::shared_ptr<Tensor> input_cpu=std::make_shared<Tensor>
            (allocator_cpu,in_shapes,DataType::kDataTypeFP32);
    std::shared_ptr<Tensor> weight_cpu=std::make_shared<Tensor>
            (allocator_cpu,w_shapes,DataType::kDataTypeFP32);
    std::shared_ptr<Tensor> output_cpu=std::make_shared<Tensor>
            (allocator_cpu,out_shapes,DataType::kDataTypeFP32);
    float *in_ptr_cpu=input_cpu->tensor_data_ptr<float>();
    float *w_ptr_cpu=weight_cpu->tensor_data_ptr<float>();
    in_ptr_cpu[0]=1;in_ptr_cpu[1]=2;in_ptr_cpu[2]=3;
    in_ptr_cpu[3]=4;in_ptr_cpu[4]=5;in_ptr_cpu[5]=6;
    w_ptr_cpu[0]=1;w_ptr_cpu[1]=2;w_ptr_cpu[2]=3;
    w_ptr_cpu[3]=4;w_ptr_cpu[4]=5;w_ptr_cpu[5]=6;
    w_ptr_cpu[6]=7;w_ptr_cpu[7]=8;w_ptr_cpu[8]=9;
    w_ptr_cpu[9]=10;w_ptr_cpu[10]=11;w_ptr_cpu[11]=12;
    matmul_tensor(input_cpu,weight_cpu,output_cpu);
    output_cpu->tensor_print_data();
    std::cout<<"-----CUDA-----\n";
    std::shared_ptr<Tensor> input_cuda=input_cpu->tensor_to_cuda(allocator_cuda);
    std::shared_ptr<Tensor> weight_cuda=weight_cpu->tensor_to_cuda(allocator_cuda);
    std::shared_ptr<Tensor> out_cuda=std::make_shared<Tensor>(allocator_cuda,out_shapes,DataType::kDataTypeFP32);
    matmul_tensor(input_cuda,weight_cuda,out_cuda);
    std::shared_ptr<Tensor> out_print=std::make_shared<Tensor>(allocator_cpu,out_shapes,DataType::kDataTypeFP32);
    out_print=out_cuda->tensor_to_cpu(allocator_cpu);
    out_print->tensor_print_data();
}