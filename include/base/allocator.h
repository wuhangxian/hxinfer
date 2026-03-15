#ifndef HXINFER_ALLOCATOR_H
#define HXINFER_ALLOCATOR_H
#include "cstring"
#include "base/config.h"

namespace hxinfer{
    class Allocator{
    public:
        virtual void* allocate(size_t byte_size)=0;
        virtual void release(void* ptr)=0;
        virtual ~Allocator()=default;
        virtual DeviceType device_type() =0;
    };

    class CPUAllocator:public Allocator{
    public:
        void* allocate(size_t byte_size) override;
        void release(void* ptr) override;
        DeviceType device_type() override{ return DeviceType::kDeviceCPU;}
    };

    class CUDAAllocator:public Allocator{
    public:
        void* allocate(size_t byte_size) override;
        void release(void* ptr) override;
        DeviceType device_type() override{return DeviceType::kDeviceCUDA;}
    };
}

#endif //HXINFER_ALLOCATOR_H
