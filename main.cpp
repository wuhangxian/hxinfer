#include <iostream>
#include "base/allocator.h" // 只要 CMake 改对了，这行就不会报红

int main() {
    std::cout << "Test Start..." << std::endl;

    // 使用工厂模式创建分配器
    auto cpu_allocator = base::CPUDeviceAllocatorFactory::get_instance();

    // 申请内存
    size_t size = 1024;
    void* ptr = cpu_allocator->allocate(size);

    if (ptr) {
        std::cout << "Allocated success: " << ptr << std::endl;

        // 模拟清零
        cpu_allocator->memset_zero(ptr, size, nullptr);

        // 释放
        cpu_allocator->release(ptr);
    }

    std::cout << "Test End." << std::endl;
    return 0;
}