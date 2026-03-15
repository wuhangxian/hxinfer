#include "op/math_ops.h"
#include "base/dispatch.h"
#include "iostream"
namespace hxinfer{
    void add_cpu(const std::shared_ptr<Tensor>& input_a,const std::shared_ptr<Tensor>& input_b,
                    std::shared_ptr<Tensor>& output){
        if(input_a->tensor_device_type()!=DeviceType::kDeviceCPU||
           input_b->tensor_device_type()!=DeviceType::kDeviceCPU||
           output->tensor_device_type()!=DeviceType::kDeviceCPU){
            std::cerr<<"[Fatal error] add_cuda expects CPU Tensors!";
        }
        if(input_a->tensor_total_elements()!=input_b->tensor_total_elements()||
           input_a->tensor_total_elements()!=output->tensor_total_elements()){
            throw std::runtime_error("[Fatal error] add_cuda的input_a与input_b与output数量大小不匹配!\n");
        }
        DataType type_a=input_a->tensor_data_type();
        DataType type_b=input_b->tensor_data_type();
        DataType type_out=output->tensor_data_type();
        if(type_a!=type_b){
            throw std::runtime_error("add_tensor的input_a与input_b数据类型不匹配!\n");
        }
        if(type_a!=type_out){
            throw std::runtime_error("add_tensor的input_a与output数据类型不匹配!\n");
        }
        if(type_b!=type_out){
            throw std::runtime_error("add_tensor的input_b与output数据类型不匹配!\n");
        }
// 🚀 核心改造：工业级的 Upcasting 计算逻辑
        auto add_logic = [](const auto* ptr_a, const auto* ptr_b, auto* ptr_out, size_t total_elements) {
            // 获取输出指针的真实物理类型 (比如 float16_t, float 等)
            // TODO 暂时只是进行了输出类型的强转,
            //  这是不完善的,对于不同类型的数据应该有对应的逻辑
            using OutType = std::decay_t<decltype(*ptr_out)>;

            for (size_t i = 0; i < total_elements; i++) {
                // 第一步：物理层 -> 数学层 (Upcasting 精度上拉)
                // 不管指针指向的是 2 字节还是 4 字节，统统安全转换为标准的 4 字节 float
                float val_a = static_cast<float>(ptr_a[i]);
                float val_b = static_cast<float>(ptr_b[i]);

                // 第二步：工作台纯计算
                // 在 FP32 域进行极其安全的数学加法，绝对不会发生 FP16 容易出现的溢出！
                float val_out = val_a + val_b;

                // 第三步：数学层 -> 物理层 (Downcasting 精度下调)
                // 将极其精确的 FP32 结果，强转回目标类型，塞回物理内存
                ptr_out[i] = static_cast<OutType>(val_out);
            }
        };
        HXINFER_DISPATCH_ALL_TYPES(type_a,"add_tensor",[&](){
            add_logic(input_a->tensor_data_ptr<scalar_t>(),
                    input_b->tensor_data_ptr<scalar_t>(),
                    output->tensor_data_ptr<scalar_t>(),
                    output->tensor_total_elements());
        });
    }

}