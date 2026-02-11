#include <cstdlib>
#include <iostream>
#include <cstring>
class Allocator{
    public:
    // virtual = 虚函数。意思是：我不干活，具体怎么干，让儿子(子类)去干。
    // = 0    = 纯虚函数。意思是：我不仅不干活，连个默认方案都没有，儿子必须重写这个函数。
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

int main(){
    Allocator* my_allocator=new CPUAllocator();
    size_t byte_size=100*sizeof( float);
    float* data=(float*)my_allocator->allocate(byte_size);
    std::cout << "第一个数是: " << data[0] << " (应该是0)" << std::endl;
    // 我们可以随便改写它
    data[0] = 999.99f;
    std::cout << "赋值后是: " << data[0] << std::endl;
    // 只要调 release，不管是 CPU 还是 GPU，它自己知道该怎么释放
    my_allocator->release(data);
    std::cout << "释放内存后是: " << data[0] << std::endl;
    //当你写下 delete my_allocator; 时，按顺序发生了两件事：
    //第一步（扣扳机）： 自动调用析构函数 (~Allocator)。
    //作用：让对象在临死前处理后事（比如关文件、断开连接）。
    //第二步（子弹飞出）： 释放对象本身的内存。
    //作用：把对象占用的那一点点内存（比如 8 字节）还给操作系统。
    delete my_allocator;
    return 0;
}

