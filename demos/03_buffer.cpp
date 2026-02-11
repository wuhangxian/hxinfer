#include <cstdlib>
#include <iostream>
#include <cstring>

class Allocator{
public:
    virtual void* allocate(size_t byte_size)=0;
    virtual void release(void* ptr)=0;
    virtual ~Allocator(){};
};
class CPUAllocator:public Allocator{
public:
    void* allocate(size_t byte_size) override{
        void* ptr= malloc(byte_size);
        if(ptr== nullptr){
            std::cerr<<"[CPU]内存申请失败!"<<std::endl;
            return nullptr;
        }
        // 帮用户把房间打扫干净 (清零)
        // memset(地皮指针, 初始值0, 大小)
        // 这样用户拿到的内存全是 0，不会有乱码
        memset(ptr,0,byte_size);
        std::cout<<"[CPU]成功申请了"<<byte_size<<"字节的内存"<<std::endl;
        return ptr;
    }

    void release(void* ptr) override{
        if(ptr!= nullptr){
            free(ptr);
            std::cout<<"[CPU]成功释放了内存"<<std::endl;
        }
    }
};
class Buffer{
private:
    void* data_;
    size_t byte_size_;
    Allocator* allocator_;

public:
    Buffer(size_t byte_size,Allocator* allocator):
    byte_size_(byte_size),
    allocator_(allocator){
        data_=allocator_->allocate(byte_size_);
    }

    ~Buffer(){
        if(data_!= nullptr){
            allocator_->release(data_);
            data_=nullptr;
            std::cout<<"Buffer被释放了"<<std::endl;
        }
    }

    void* data(){
        return data_;
    }
};
int main(){
    Allocator* allocator=new CPUAllocator();
    {
        std::cout << "...开始测试..." << std::endl;
        Buffer buffer(100 * sizeof(float), allocator);
        float* ptr=(float*)buffer.data();
        ptr[0] = 123.456f;
        std::cout << "数据: " << ptr[0] << std::endl;
        std::cout << "--- 作用域即将结束，buffer 要死了 ---" << std::endl;
    }
    std::cout << "--- 作用域结束，内存已经被自动释放了 ---" << std::endl;
    delete allocator;
    return 0;
}

