#include "base/buffer.h"
namespace hxinfer{
    Buffer::Buffer(std::shared_ptr<Allocator> allocator, size_t byte_size):
            allocator_(allocator),byte_size_(byte_size){
        data_=allocator_->allocate(byte_size_);
    }
    Buffer::~Buffer() {
        if(data_== nullptr){
            allocator_->release(data_);
            data_= nullptr;
        }
    }
}
