#include "tensor/tensor.h"
#include "base/dispatch.h"
#include "iostream"
namespace hxinfer{
    void matmul_cpu(const std::shared_ptr<Tensor>& input,const std::shared_ptr<Tensor>& weight,
                       std::shared_ptr<Tensor>& output){
        if(input->tensor_device_type()!=DeviceType::kDeviceCPU||
           weight->tensor_device_type()!=DeviceType::kDeviceCPU||
           output->tensor_device_type()!=DeviceType::kDeviceCPU){
            std::cerr<<"[Fatal Error] matmul_cuda expects all tensors to be on CPU"<<std::endl;
            return;
        }
        DataType type_out=output->tensor_data_type();
        DataType type_in = input->tensor_data_type();
        if (type_in != weight->tensor_data_type() || type_in != type_out) {
            throw std::runtime_error("matmul_tensor: input, weight, output 的数据类型必须完全一致!\n");
        }
        auto matmul_logic=[&](const auto* ptr_in,
                const auto*ptr_weight,auto *ptr_out){
            using OutType=std::decay_t<decltype(*ptr_out)>;
            std::vector<int> in_shapes=input->tensor_shapes();
            int K=in_shapes[in_shapes.size()-1];
            size_t M=input->tensor_total_elements()/K;
            size_t N=weight->tensor_total_elements()/K;
            // 🚀 补上最后一道安检门：绝不写穿内存！
            if (output->tensor_total_elements() != M * N) {
                throw std::runtime_error("matmul_tensor: 致命错误，输出张量的内存空间与 M*N 不匹配！");
            }
            for(size_t i=0;i<M;i++){
                auto *cur_out=ptr_out+i*N;
                auto *curr_in=ptr_in+i*K;
                for(size_t j=0;j<N;j++){
                    auto *curr_weight=ptr_weight+j*K;
                    float sum=0;
                    for(int k=0;k<K;k++){
                        sum=sum+static_cast<float >(curr_in[k])*static_cast<float >(curr_weight[k]);
                    }
                    cur_out[j]=static_cast<OutType>(sum);
                }
            }
        };
        HXINFER_DISPATCH_ALL_TYPES(type_out,"matmul",[&](){
            matmul_logic(input->tensor_data_ptr<scalar_t>(),
                    weight->tensor_data_ptr<scalar_t>(),
                    output->tensor_data_ptr<scalar_t>());
        });
    }
}

