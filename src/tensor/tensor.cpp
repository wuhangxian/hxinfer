#include "tensor/tensor.h"
#include "iostream"
namespace hxinfer{
    Tensor::Tensor(std::shared_ptr<Allocator> allocator, std::vector<int> shapes, DataType data_type)
            :shapes_(shapes),data_type_(data_type){
        total_elements_=std::accumulate(shapes_.begin(),shapes_.end(),
                                        size_t{1},std::multiplies<>());
        total_byte_size_=total_elements_* DataTypeSize(data_type_);
        buffer_=std::make_shared<Buffer>(allocator,total_byte_size_);
    }

    void Tensor::tensor_print_data() {
        auto print_logic = [](const auto *p,std::vector<int> shapes) {
            int n = shapes.size();
            if (n == 1) {
                int len = shapes[0];
                std::cout << '[';
                for (int i = 0; i < len; i++) {
                    std::cout << static_cast<float >(p[i]) << ' ';
                }
                std::cout << ']' << '\n';
            }
            if (n == 2) {
                int row = shapes[0];
                int col = shapes[1];
                std::cout << '[' << '\n';
                for (int i = 0; i < row; i++) {
                    std::cout << '[';
                    const auto *curr_p = p + i * col;
                    for (int j = 0; j < col; j++) {
                        std::cout << static_cast<float>(curr_p[j]) << ' ';
                    }
                    std::cout << ']' << '\n';
                }
                std::cout << ']' << '\n';
            }
            if (n == 3) {
                int space = shapes[0];
                int row = shapes[1];
                int col = shapes[2];
                std::cout << '[' << '\n';
                for (int i = 0; i < space; i++) {
                    std::cout << '[' << '\n';
                    for (int j = 0; j < row; j++) {
                        std::cout << '[';
                        // 同理，用 auto 替代 const float*
                        const auto *curr_p = p + i * row * col + j * col;
                        for (int k = 0; k < col; k++) {
                            std::cout << static_cast<float>(curr_p[k]) << ' ';
                        }
                        std::cout << ']' << '\n';
                    }
                    std::cout << ']' << '\n';
                }
                std::cout << ']' << '\n';
            }
        };
        switch (data_type_) {
            case DataType::kDataTypeFP32:
                print_logic(tensor_data_ptr<float>(),
                        shapes_); // 编译器提取 float*，并生成一份 float 版的 print_logic
                break;
            case DataType::kDataTypeInt8:
                print_logic(tensor_data_ptr<int8_t>(),
                        shapes_); // 编译器提取 int8_t*，再生成一份 int8_t 版的 print_logic
                break;
            default:
                throw std::runtime_error("不支持的 DataType 打印!");
        }
    }
}

