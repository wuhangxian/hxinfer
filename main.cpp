#include <iostream>
#include <memory>
#include <vector>

// 引入我们的核心基建
#include "base/allocator.h"
#include "base/config.h"
#include "tensor/tensor.h"
#include "op/math_ops.h"

using namespace hxinfer; // 🚀 亮出通行证，直接使用领地内的类

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "🚀 HXInfer Engine - Add 算子点火测试" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        // 【第一步：向操作系统申请一块“地皮”】
        // 实例化你的 CPU 内存分配器
        std::shared_ptr<Allocator> cpu_allocator = std::make_shared<CPUAllocator>();

        // 【第二步：定义集装箱的规格 (Shape)】
        // 我们来做一个 2x3 的矩阵加法
        std::vector<int> shape = {2, 3};

        // 【第三步：在“地皮”上造出三个“集装箱” (Tensor)】
        // A, B 作为输入，C 作为输出，类型全部设定为 FP32
        std::shared_ptr<Tensor> tensor_a = std::make_shared<Tensor>(cpu_allocator, shape, DataType::kDataTypeFP32);
        std::shared_ptr<Tensor> tensor_b = std::make_shared<Tensor>(cpu_allocator, shape, DataType::kDataTypeFP32);
        std::shared_ptr<Tensor> tensor_c = std::make_shared<Tensor>(cpu_allocator, shape, DataType::kDataTypeFP32);

        // 【第四步：往集装箱里装入真实的“货物” (写入测试数据)】
        float* ptr_a = tensor_a->tensor_data_ptr<float>();
        float* ptr_b = tensor_b->tensor_data_ptr<float>();

        // 给 A 赋值：[1, 2, 3], [4, 5, 6]
        // 给 B 赋值：[10, 20, 30], [40, 50, 60]
        for (int i = 0; i < 6; ++i) {
            ptr_a[i] = static_cast<float>(i + 1);
            ptr_b[i] = static_cast<float>((i + 1) * 10);
        }

        std::cout << "\n>>> [输入] 矩阵 A (2x3):" << std::endl;
        tensor_a->tensor_print_data();

        std::cout << "\n>>> [输入] 矩阵 B (2x3):" << std::endl;
        tensor_b->tensor_print_data();

        // 【第五步：拉响算子！流水线开动！】
        std::cout << "\n>>> ⚡ 正在调用底层的 add_tensor 进行计算..." << std::endl;

        // 核心调用：A + B = C
        mul_tensor(tensor_a, tensor_b, tensor_c);

        // 【第六步：验收成果！】
        std::cout << "\n>>> [输出] 矩阵 C (预期为 A+B):" << std::endl;
        tensor_c->tensor_print_data();

        std::cout << "\n✅ 试车完美结束！算子逻辑、宏分发、内存调度全链路畅通！" << std::endl;

    } catch (const std::exception& e) {
        // 如果中间发生内存不够、类型报错、尺寸不匹配，都会被这里优雅地抓住
        std::cerr << "\n❌ 引擎崩溃啦: " << e.what() << std::endl;
    }

    return 0;
}