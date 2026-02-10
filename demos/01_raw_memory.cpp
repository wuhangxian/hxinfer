#include "iostream"

void debug_print(const char* name,float* ptr,int size){
    std::cout<<name<<"[";
    for(int i=0;i<std::min(5, size);i++){
        std::cout<<ptr[i]<<",";
    }
    std::cout<<"...]"<<std::endl;
}

int main(){
    int length=100;
    size_t bytes=length*sizeof(float);
    std::cout<<"准备申请内存"<<bytes<<"字节"<<std::endl;
    float* ptr_a = (float*)malloc(bytes);
    float* ptr_b = (float*)malloc(bytes);
    float* ptr_c = (float*)malloc(bytes); // 用来存结果
    if (ptr_a == nullptr || ptr_b == nullptr || ptr_c == nullptr) {
        std::cerr << "内存申请失败！" << std::endl;
        return -1;
    }
    // ==========================================
    // 2. 初始化数据 (Initialization)
    // ==========================================
    // 这就是在模拟 Tensor 的赋值
    for (int i = 0; i < length; ++i) {
        ptr_a[i] = 1.0f;      // A 全是 1.0
        ptr_b[i] = (float)i;  // B 是 0, 1, 2, 3...
    }

    // ==========================================
    // 3. 计算 (Compute / Operator)
    // ==========================================
    // 这就是在模拟 Add 算子！
    // 也就是 Tensor::ptr() 获取指针后做的事情
    for (int i = 0; i < length; ++i) {
        ptr_c[i] = ptr_a[i] + ptr_b[i];
    }
    // ==========================================
    // 4. 验证结果
    // ==========================================
    debug_print("Input A", ptr_a, length);
    debug_print("Input B", ptr_b, length);
    debug_print("Output C", ptr_c, length);

    // ==========================================
    // 5. 释放地皮 (Deallocation) - 最重要的一步！
    // ==========================================
    // 如果没有这一步，就是 "内存泄漏 (Memory Leak)"
    // 这就是在模拟 Allocator::free
    free(ptr_a);
    free(ptr_b);
    free(ptr_c);

    // 避免悬空指针 (Dangling Pointer)
    // 释放后，指针里的地址还在，但那个地址已经不归你管了
    ptr_a = nullptr;
    ptr_b = nullptr;
    ptr_c = nullptr;

    std::cout << "内存已释放，程序安全退出。" << std::endl;
    return 0;
}

