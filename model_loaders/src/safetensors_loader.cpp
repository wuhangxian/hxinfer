#include "safetensors_loader.h"
#include <iostream>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace hxinfer {

using json = nlohmann::json;

// BF16 -> FP16 conversion (bit manipulation, no float arithmetic)
static void bf16_to_fp16(const uint16_t* src, __half* dst, size_t n) {
    // BF16: [sign(1)] [exponent(8)] [mantissa(7)]
    // FP16: [sign(1)] [exponent(5)] [mantissa(10)]
    // BF16 exponent is 127-biased, FP16 is 15-biased, diff = 112
    // Simplified: shift left by 3 to get FP16 mantissa alignment,
 // but need exponent adjustment
    // Actually: BF16 is just the top 16 bits of FP32.
    // FP16 is a different format. We need proper conversion.
    // Simplest correct approach: BF16 -> FP32 -> FP16
    for (size_t i = 0; i < n; i++) {
        uint32_t f32_bits = ((uint32_t)src[i]) << 16;  // BF16 -> FP32 (zero-extend mantissa)
        float f32_val;
        std::memcpy(&f32_val, &f32_bits, 4);
        dst[i] = __float2half(f32_val);
    }
}

SafetensorsReader::~SafetensorsReader() {
    if (mmap_base_ && mmap_base_ != MAP_FAILED) {
        munmap(mmap_base_, file_size_);
    }
    if (fd_ >= 0) close(fd_);
}

bool SafetensorsReader::load(const std::string& file_path) {
    fd_ = open(file_path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        std::cerr << "[SafetensorsReader] Cannot open: " << file_path << std::endl;
        return false;
    }

    struct stat sb;
    if (fstat(fd_, &sb) < 0) {
        std::cerr << "[SafetensorsReader] fstat failed" << std::endl;
        return false;
    }
    file_size_ = sb.st_size;

    mmap_base_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mmap_base_ == MAP_FAILED) {
        std::cerr << "[SafetensorsReader] mmap failed" << std::endl;
        return false;
    }

    uint64_t header_len;
    std::memcpy(&header_len, mmap_base_, 8);

    const char* header_start = static_cast<const char*>(mmap_base_) + 8;
    json header = json::parse(header_start, header_start + header_len);

    for (auto& [key, val] : header.items()) {
        if (key == "__metadata__") continue;

        SafetensorInfo info;
        info.dtype = val["dtype"].get<std::string>();
        for (auto& dim : val["shape"]) {
            info.shape.push_back(dim.get<int>());
        }
        info.offset_begin = val["data_offsets"][0].get<size_t>();
        info.offset_end   = val["data_offsets"][1].get<size_t>();
        tensor_map_[key] = info;
    }

    size_t data_start = 8 + header_len;
    for (auto& [key, info] : tensor_map_) {
        info.offset_begin += data_start;
        info.offset_end   += data_start;
    }

    return true;
}

bool SafetensorsReader::has_tensor(const std::string& name) const {
    return tensor_map_.count(name) > 0;
}

const SafetensorInfo* SafetensorsReader::get_info(const std::string& name) const {
    auto it = tensor_map_.find(name);
    if (it == tensor_map_.end()) return nullptr;
    return &it->second;
}

const void* SafetensorsReader::get_raw_ptr(const std::string& name) const {
    auto it = tensor_map_.find(name);
    if (it == tensor_map_.end()) return nullptr;
    return static_cast<const char*>(mmap_base_) + it->second.offset_begin;
}

DataType SafetensorsReader::parse_dtype(const std::string& dtype_str) {
    if (dtype_str == "F32") return DataType::kDataTypeFP32;
    if (dtype_str == "F16") return DataType::kDataTypeFP16;
    if (dtype_str == "BF16") return DataType::kDataTypeFP16;  // will convert
    if (dtype_str == "I8") return DataType::kDataTypeInt8;
    return DataType::kDataTypeFP32;
}

std::shared_ptr<Tensor> SafetensorsReader::get_tensor_gpu(
    const std::string& name,
    const std::shared_ptr<CUDAAllocator>& alloc,
    DataType target_dtype) const
{
    auto it = tensor_map_.find(name);
    if (it == tensor_map_.end()) {
        std::cerr << "[SafetensorsReader] Tensor not found: " << name << std::endl;
        return nullptr;
    }

    const auto& info = it->second;
    const void* src = static_cast<const char*>(mmap_base_) + info.offset_begin;
    size_t n = 1;
    for (auto d : info.shape) n *= d;

    // BF16 source -> FP16 target: convert on CPU then upload
    if (info.dtype == "BF16" && target_dtype == DataType::kDataTypeFP16) {
        std::vector<__half> f16_buf(n);
        bf16_to_fp16(static_cast<const uint16_t*>(src), f16_buf.data(), n);
        auto t = std::make_shared<Tensor>(alloc, info.shape, DataType::kDataTypeFP16);
        t->tensor_set_device_type(DeviceType::kDeviceCUDA);
        cudaMemcpy(t->raw_data_ptr(), f16_buf.data(), n * sizeof(__half), cudaMemcpyHostToDevice);
        return t;
    }

    // BF16 source -> FP32 target: convert
    if (info.dtype == "BF16" && target_dtype == DataType::kDataTypeFP32) {
        std::vector<float> f32_buf(n);
        const uint16_t* bf16_ptr = static_cast<const uint16_t*>(src);
        for (size_t i = 0; i < n; i++) {
            uint32_t bits = ((uint32_t)bf16_ptr[i]) << 16;
            std::memcpy(&f32_buf[i], &bits, 4);
        }
        auto t = std::make_shared<Tensor>(alloc, info.shape, DataType::kDataTypeFP32);
        t->tensor_set_device_type(DeviceType::kDeviceCUDA);
        cudaMemcpy(t->raw_data_ptr(), f32_buf.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        return t;
    }

    // F16 source -> FP16 target: direct copy
    if (info.dtype == "F16" && target_dtype == DataType::kDataTypeFP16) {
        size_t nbytes = n * sizeof(__half);
        auto t = std::make_shared<Tensor>(alloc, info.shape, DataType::kDataTypeFP16);
        t->tensor_set_device_type(DeviceType::kDeviceCUDA);
        cudaMemcpy(t->raw_data_ptr(), src, nbytes, cudaMemcpyHostToDevice);
        return t;
    }

    // F32 source -> FP16 target: convert
    if (info.dtype == "F32" && target_dtype == DataType::kDataTypeFP16) {
        std::vector<__half> f16_buf(n);
        const float* f32_ptr = static_cast<const float*>(src);
        for (size_t i = 0; i < n; i++) {
            f16_buf[i] = __float2half(f32_ptr[i]);
        }
        auto t = std::make_shared<Tensor>(alloc, info.shape, DataType::kDataTypeFP16);
        t->tensor_set_device_type(DeviceType::kDeviceCUDA);
        cudaMemcpy(t->raw_data_ptr(), f16_buf.data(), n * sizeof(__half), cudaMemcpyHostToDevice);
        return t;
    }

    // F32 source -> FP32 target: direct copy
    if (info.dtype == "F32" && target_dtype == DataType::kDataTypeFP32) {
        size_t nbytes = n * sizeof(float);
        auto t = std::make_shared<Tensor>(alloc, info.shape, DataType::kDataTypeFP32);
        t->tensor_set_device_type(DeviceType::kDeviceCUDA);
        cudaMemcpy(t->raw_data_ptr(), src, nbytes, cudaMemcpyHostToDevice);
        return t;
    }

    // F16 source -> FP32 target: convert
    if ((info.dtype == "F16" || info.dtype == "BF16") && target_dtype == DataType::kDataTypeFP32) {
        std::vector<float> f32_buf(n);
        if (info.dtype == "BF16") {
            const uint16_t* bf16_ptr = static_cast<const uint16_t*>(src);
            for (size_t i = 0; i < n; i++) {
                uint32_t bits = ((uint32_t)bf16_ptr[i]) << 16;
                std::memcpy(&f32_buf[i], &bits, 4);
            }
        } else {
            const __half* f16_ptr = static_cast<const __half*>(src);
            for (size_t i = 0; i < n; i++) {
                f32_buf[i] = __half2float(f16_ptr[i]);
            }
        }
        auto t = std::make_shared<Tensor>(alloc, info.shape, DataType::kDataTypeFP32);
        t->tensor_set_device_type(DeviceType::kDeviceCUDA);
        cudaMemcpy(t->raw_data_ptr(), f32_buf.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        return t;
    }

    // Fallback: direct copy
    size_t nbytes = info.offset_end - info.offset_begin;
    auto t = std::make_shared<Tensor>(alloc, info.shape, target_dtype);
    t->tensor_set_device_type(DeviceType::kDeviceCUDA);
    cudaMemcpy(t->raw_data_ptr(), src, nbytes, cudaMemcpyHostToDevice);
    return t;
}

// ==================== Multi-shard reader ====================

bool SafetensorsMultiReader::load_directory(const std::string& dir_path) {
    std::string cmd = "ls " + dir_path + "/*.safetensors 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return false;

    char buf[4096];
    std::vector<std::string> files;
    while (fgets(buf, sizeof(buf), pipe)) {
        std::string f(buf);
        while (!f.empty() && f.back() == '\n') f.pop_back();
        files.push_back(f);
    }
    pclose(pipe);

    if (files.empty()) {
        std::string single = dir_path + "/model.safetensors";
        files.push_back(single);
    }

    for (const auto& f : files) {
        auto reader = std::make_unique<SafetensorsReader>();
        if (reader->load(f)) {
            readers_.push_back(std::move(reader));
        }
    }

    return !readers_.empty();
}

bool SafetensorsMultiReader::has_tensor(const std::string& name) const {
    for (const auto& r : readers_) {
        if (r->has_tensor(name)) return true;
    }
    return false;
}

std::shared_ptr<Tensor> SafetensorsMultiReader::get_tensor_gpu(
    const std::string& name,
    const std::shared_ptr<CUDAAllocator>& alloc,
    DataType target_dtype) const
{
    for (const auto& r : readers_) {
        if (r->has_tensor(name)) {
            return r->get_tensor_gpu(name, alloc, target_dtype);
        }
    }
    std::cerr << "[SafetensorsMultiReader] Tensor not found: " << name << std::endl;
    return nullptr;
}

size_t SafetensorsMultiReader::num_tensors() const {
    size_t total = 0;
    for (const auto& r : readers_) total += r->num_tensors();
    return total;
}

std::vector<std::string> SafetensorsMultiReader::tensor_names() const {
    return {};
}

} // namespace hxinfer
