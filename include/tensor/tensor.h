#ifndef HXINFER_TENSOR_H
#define HXINFER_TENSOR_H
#include "cstring"
#include "memory"
#include "base/buffer.h"
#include "base/allocator.h"
#include "base/config.h"
#include "vector"
#include "numeric"
#include "stdexcept"

namespace hxinfer{
    class Tensor{
    private:
        std::vector<int> shapes_;
        DataType data_type_;
        size_t total_elements_;
        size_t total_byte_size_;
        std::shared_ptr<Buffer> buffer_;
    public:
        Tensor(std::shared_ptr<Allocator> allocator,std::vector<int> shapes,DataType data_type);
        ~Tensor()=default;
        const std::vector<int>& tensor_shapes() const{return shapes_;}
        DataType tensor_data_type() const{return data_type_;}
        size_t tensor_total_elements() const{return total_elements_;}
        size_t tensor_total_byte_size() const{return total_byte_size_;}
        void tensor_reshape(std::vector<int> new_shapes){
            size_t new_total_elements=std::accumulate(new_shapes.begin(),new_shapes.end(),size_t{1},
                                                      std::multiplies<>());
            size_t new_total_byte_size=new_total_elements* DataTypeSize(data_type_);
            if(new_total_byte_size>total_byte_size_){
                throw std::runtime_error("内存大小不够!不可以进行reshape\n");
            }else{
                total_elements_=new_total_elements;
                shapes_=new_shapes;
            }
        }

        template<typename T>
        const T* tensor_data_ptr() const{
            return static_cast<const T*>(buffer_->buffer_data_ptr());
        }
        template<typename T>
        T* tensor_data_ptr(){
            return static_cast<T*>(buffer_->buffer_data_ptr());
        }
        void tensor_print_data();
    };
}

#endif //HXINFER_TENSOR_H
