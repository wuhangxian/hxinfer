// demos/02_allocator_test.cu
#include "base/allocator.h"
#include <iostream>
#include <memory>

using namespace hxinfer;

int main() {
    std::cout << "--- Testing CPU Allocator ---" << std::endl;
    std::shared_ptr<Allocator> cpu_alloc = std::make_shared<CPUAllocator>();

    void* cpu_ptr = cpu_alloc->allocate(1024); // 在 CPU 分配 1KB
    if (cpu_ptr) {
        std::cout << "CPU Memory Allocated at: " << cpu_ptr << std::endl;
        cpu_alloc->release(cpu_ptr);
        std::cout << "CPU Memory Released." << std::endl;
    }

    std::cout << "\n--- Testing CUDA Allocator ---" << std::endl;
    std::shared_ptr<Allocator> cuda_alloc = std::make_shared<CUDAAllocator>();

    void* cuda_ptr = cuda_alloc->allocate(1024 * 1024 * 100); // 在显卡分配 100MB
    if (cuda_ptr) {
        std::cout << "CUDA Memory Allocated at: " << cuda_ptr << std::endl;
        cuda_alloc->release(cuda_ptr);
        std::cout << "CUDA Memory Released." << std::endl;
    }

    return 0;
}