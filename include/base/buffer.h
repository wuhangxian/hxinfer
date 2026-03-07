#ifndef HXINFER_BUFFER_H
#define HXINFER_BUFFER_H
#include "memory"
#include "cstring"
#include "allocator.h"
namespace hxinfer{
    class Buffer{
    private:
        std::shared_ptr<Allocator> allocator_;
        size_t byte_size_;
        void* data_;
    public:
        Buffer(std::shared_ptr<Allocator> allocator,size_t byte_size);
        ~Buffer();
        size_t buffer_byte_size() const { return byte_size_; }
        const void* buffer_data_ptr() const { return data_; }
        void* buffer_data_ptr() { return data_; }
    };
}


#endif //HXINFER_BUFFER_H
