#include "base/allocator.h"
#include "cstring"
#include "cuda_runtime.h"
#include "iostream"
namespace hxinfer{
    void* CUDAAllocator::allocate(size_t byte_size){
        if(byte_size==0){
            return nullptr;
        }
        void* ptr= nullptr;
        cudaError_t err=cudaMalloc(&ptr,byte_size);
        if(err!=cudaSuccess){
            std::cerr<<"[Fatal Error]CUDA Malloc Fail!"
                     <<"Reason:"<<cudaGetErrorString(err)
                     <<"| Requested Bytes: "<<byte_size<<std::endl;
            return nullptr;
        }
        return ptr;
    }

    void CUDAAllocator::release(void *ptr) {
        if(ptr!= nullptr){
            cudaFree(ptr);
        }
    }
}