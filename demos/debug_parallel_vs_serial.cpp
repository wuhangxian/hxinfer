#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <cstdlib>
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "base/allocator.h"
#include "base/config.h"
#include "tensor/tensor.h"
#include "op/math_ops.h"
#include "layer/attention.h"
#include "layer/transformer.h"
#include "model/causal_lm_model.h"
#include "llama_weight_loader.h"
#include "llama7b_tokenizer.h"
using namespace hxinfer;
int main(){
    auto cpu = std::make_shared<CPUAllocator>();
    auto cu = std::make_shared<CUDAAllocator>();
    ModelConfig config;
    auto model = LlamaWeightLoader::load("/root/dorianwu/models/llama2-7b/llama2-7b-safetensors", config, cpu, cu);
    // Get layer 0 attention
    auto& blocks = model->get_blocks();
    auto& attn = blocks[0]->get_attention();
    // Create input [4, dim] on GPU
    int S = 4;
    int dim = config.dim;
    auto in = std::make_shared<Tensor>(cu, std::vector<int>{S, dim}, DataType::kDataTypeFP16);
    in->tensor_set_device_type(DeviceType::kDeviceCUDA);
    // Fill with known values
    std::vector<__half> h_in(S * dim);
    for(int i=0;i<S*dim;i++) h_in[i] = __float2half(0.5f);
    cudaMemcpy(in->raw_data_ptr(), h_in.data(), S*dim*sizeof(__half), cudaMemcpyHostToDevice);
    // Call forward_prefill
    auto out = std::make_shared<Tensor>(cu, std::vector<int>{S, dim}, DataType::kDataTypeFP16);
    out->tensor_set_device_type(DeviceType::kDeviceCUDA);
    attn->forward_prefill(in, out, S);
    cudaDeviceSynchronize();
    // Get output
    std::vector<__half> h_out(S * dim);
    cudaMemcpy(h_out.data(), out->raw_data_ptr(), S*dim*sizeof(__half), cudaMemcpyDeviceToHost);
    // Print first few values
    std::cout << "Parallel attention output:" << std::endl;
    for(int i=0;i<10;i++) std::cout << __half2float(h_out[i]) << " ";
    std::cout << std::endl;
    // Now do serial: forward() 4 times
    auto in1 = std::make_shared<Tensor>(cu, std::vector<int>{1, dim}, DataType::kDataTypeFP16);
    in1->tensor_set_device_type(DeviceType::kDeviceCUDA);
    auto out1 = std::make_shared<Tensor>(cu, std::vector<int>{1, dim}, DataType::kDataTypeFP16);
    out1->tensor_set_device_type(DeviceType::kDeviceCUDA);
    std::cout << "Serial attention output:" << std::endl;
    for(int s=0;s<S;s++){
        cudaMemcpy(in1->raw_data_ptr(),
                   static_cast<char*>(in->raw_data_ptr()) + (size_t)s*dim*sizeof(__half),
                   dim*sizeof(__half), cudaMemcpyDeviceToDevice);
        attn->forward(in1, out1, s);
        cudaDeviceSynchronize();
        std::vector<__half> h_out1(dim);
        cudaMemcpy(h_out1.data(), out1->raw_data_ptr(), dim*sizeof(__half), cudaMemcpyDeviceToHost);
        std::cout << "  token " << s << ": ";
        for(int i=0;i<5;i++) std::cout << __half2float(h_out1[i]) << " ";
        std::cout << std::endl;
    }
    // Compare parallel token 0 with serial token 0
    std::cout << "\nCompare token 0:" << std::endl;
    std::cout << "  parallel: ";
    for(int i=0;i<5;i++) std::cout << __half2float(h_out[i]) << " ";
    std::cout << std::endl;
    // Serial token 0
    cudaMemcpy(in1->raw_data_ptr(), in->raw_data_ptr(), dim*sizeof(__half), cudaMemcpyDeviceToDevice);
    attn->forward(in1, out1, 0);
    cudaDeviceSynchronize();
    std::vector<__half> h_out1(dim);
    cudaMemcpy(h_out1.data(), out1->raw_data_ptr(), dim*sizeof(__half), cudaMemcpyDeviceToHost);
    std::cout << "  serial:   ";
    for(int i=0;i<5;i++) std::cout << __half2float(h_out1[i]) << " ";
    std::cout << std::endl;
    return 0;
}
