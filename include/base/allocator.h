#ifndef HXINFER_ALLOCATOR_H
#define HXINFER_ALLOCATOR_H
#include "cstring"

namespace hxinfer{
    class Allocator{
    public:
        virtual void* allocate(size_t byte_size)=0;
        virtual void release(void* ptr)=0;
        virtual ~Allocator()=default;
    };

    class CPUAllocator:public Allocator{
    public:
        void* allocate(size_t byte_size) override;
        void release(void* ptr) override;
    };

    class CUDAAllocator:public Allocator{
    public:
        void* allocate(size_t byte_size) override;
        void release(void* ptr) override;
    };
}

#endif //HXINFER_ALLOCATOR_H
