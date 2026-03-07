#ifndef HXINFER_DISPATCH_H
#define HXINFER_DISPATCH_H

#include <stdexcept>
#include <string>
#include "base/config.h" // 必须引入，因为宏里用到了 DataType

// 🚀 工业级神级宏：HXINFER_DISPATCH_ALL_TYPES
// 只要 include 了这个头文件，整个推理引擎任何地方都可以一键完成类型分发！
#define HXINFER_DISPATCH_ALL_TYPES(data_type, op_name, ...) \
    switch (data_type) { \
        case DataType::kDataTypeFP32: { \
            using scalar_t = float; \
            __VA_ARGS__(); \
            break; \
        } \
        case DataType::kDataTypeInt8: { \
            using scalar_t = int8_t; \
            __VA_ARGS__(); \
            break; \
        } \
        default: \
            throw std::runtime_error(std::string(op_name) + " 遇到了不支持的数据类型!"); \
    }

#endif //HXINFER_DISPATCH_H
