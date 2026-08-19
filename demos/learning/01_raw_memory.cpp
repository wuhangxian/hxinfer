#include<iostream>


void demo_print(char* name,float* data,int size){
    std::cout<<name <<"[";
    for(int i=0;i<std::min(size,10);i++){
        std::cout<<data[i]<<",";
    }
    std::cout<<"]"<<std::endl;
}
int main(){
    int length=100;
    size_t byte_size=length*sizeof(float);
    std::cout<<"准备申请内存"<<byte_size<<"字节内存"<<std::endl;
    float* ptr_a=(float*)malloc(byte_size);
    float* ptr_b=(float*)malloc(byte_size);
    float* ptr_c=(float*)malloc(byte_size);
    std::cout<<"申请内存完毕"<<std::endl;
    if(ptr_a== nullptr||ptr_b== nullptr||ptr_c== nullptr){
        std::cerr<<"申请内存失败"<<std::endl;
        return -1;
    }
    for(int i=0;i<length;i++){
        ptr_a[i]=1;
        ptr_b[i]=i;
    }
    for(int i=0;i<length;i++){
        ptr_c[i]=ptr_a[i]+ptr_b[i];
    }
    demo_print("a数组",ptr_a,length);
    demo_print("b数组",ptr_b,length);
    demo_print("c数组",ptr_c,length);
    free(ptr_a);
    free(ptr_b);
    free(ptr_c);
    return 0;
}
