#ifndef HXINFER_SAFETENSORS_LOADER_H
#define HXINFER_SAFETENSORS_LOADER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstdint>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <nlohmann/json.hpp>
#include "tensor/tensor.h"
#include "base/allocator.h"

namespace hxinfer {

struct SafetensorInfo {
    std::string dtype;
    std::vector<int> shape;
    size_t offset_begin;
    size_t offset_end;
};

class SafetensorsReader {
private:
    void* mmap_base_ = nullptr;
    size_t file_size_ = 0;
    int fd_ = -1;
    std::unordered_map<std::string, SafetensorInfo> tensor_map_;

public:
    SafetensorsReader() = default;
    ~SafetensorsReader();

    bool load(const std::string& file_path);
    bool has_tensor(const std::string& name) const;
    const SafetensorInfo* get_info(const std::string& name) const;
    size_t num_tensors() const { return tensor_map_.size(); }

    const void* get_raw_ptr(const std::string& name) const;

    std::shared_ptr<Tensor> get_tensor_gpu(
        const std::string& name,
        const std::shared_ptr<CUDAAllocator>& alloc,
        DataType target_dtype = DataType::kDataTypeFP16) const;

private:
    static DataType parse_dtype(const std::string& dtype_str);
};

class SafetensorsMultiReader {
private:
    std::vector<std::unique_ptr<SafetensorsReader>> readers_;
    std::unordered_map<std::string, int> tensor_to_shard_;

public:
    bool load_directory(const std::string& dir_path);
    bool has_tensor(const std::string& name) const;
    std::shared_ptr<Tensor> get_tensor_gpu(
        const std::string& name,
        const std::shared_ptr<CUDAAllocator>& alloc,
        DataType target_dtype = DataType::kDataTypeFP16) const;
    size_t num_tensors() const;
    std::vector<std::string> tensor_names() const;
};

} // namespace hxinfer
#endif // HXINFER_SAFETENSORS_LOADER_H
