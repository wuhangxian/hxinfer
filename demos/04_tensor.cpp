#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>
#include <numeric>

class Allocator{
public:
    virtual void* allocate(size_t byte_size)=0;
    virtual void release(void* ptr)=0;
    virtual ~Allocator(){};
};

class CPUAllocator:public Allocator{
public:
    void* allocate(size_t byte_size){
        void* ptr= malloc(byte_size);
        if(ptr== nullptr) return nullptr;
        memset(ptr,0,byte_size);
        return ptr;
    }

    void release(void* ptr){
        if(ptr!= nullptr){
            free(ptr);
        }
    }

};

class Buffer{
private:
    void* data_;
    size_t byte_size_;
    std::shared_ptr<Allocator> allocator_;

public:
    Buffer(size_t size,std::shared_ptr<Allocator> allocator)
    :byte_size_(size),allocator_(allocator){
        data_=allocator_->allocate(byte_size_);
    }

    ~Buffer(){
        if(data_!= nullptr){
            allocator_->release(data_);
            //冗余设计防止程序多次析构报错
            data_= nullptr;
        }
    }

    void* data(){
        return data_;
    }

};

class Tensor{
private:
    std::vector<int> shapes;
    std::shared_ptr<Buffer> buffer_;

public:
    Tensor(std::vector<int> shapes,std::shared_ptr<Allocator> allocator):
        shapes(shapes){
        //1. 起始位置 2. 结束位置 3. 初始值 (非常关键！) 4. 运算规则 (乘法)
        int total_elements=std::accumulate
                (shapes.begin(),shapes.end(),1,std::multiplies<int>());
        size_t byte_size=total_elements*sizeof(float);
        buffer_=std::make_shared<Buffer>(byte_size,allocator);
    }

    float& at(int index){
        float* ptr=(float*)buffer_->data();
        return ptr[index];
    }

    void print_shapes(){
        std::cout<<"该Tensor类的shapes"<<std::endl;
        for(int num:shapes){
            std::cout<<num<<" ";
        }
        std::cout<<std::endl;
    }
};

int main(){
    std::shared_ptr<Allocator> allocator=std::make_shared<CPUAllocator>();
    Tensor matrix(std::vector<int>{2,3},allocator);
    matrix.print_shapes();

    matrix.at(0)=1.1f;
    matrix.at(5)=6.6f;

    std::cout << "第一个数: " << matrix.at(0) << std::endl;
    std::cout << "最后一个数: " << matrix.at(5) << std::endl;

    return 0;
}


