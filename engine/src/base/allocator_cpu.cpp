#include "base/allocator.h"
#include "cstring"
#include "cstdlib"
#include "new"
namespace hxinfer{
    void* CPUAllocator::allocate(size_t byte_size){
        void* ptr= nullptr;
        int ret= posix_memalign(&ptr,64,byte_size);
        if(ret!=0||ptr== nullptr){
            throw std::bad_alloc();
        }
        memset(ptr,0,byte_size);
        return ptr;
    }

    void CPUAllocator::release(void *ptr) {
        if(ptr!= nullptr){
            free(ptr);
        }
    }
}
