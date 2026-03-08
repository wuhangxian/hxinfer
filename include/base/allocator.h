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
        void* allocate(size_t byte_size);
        void release(void* ptr);
    };
}

#endif //HXINFER_ALLOCATOR_H
