#include "filesystem"
#include "iostream"
#include "fstream"
#include "fcntl.h"
#include "sys/stat.h"
#include "unistd.h"
#include "sys/mman.h"


#pragma pack(push,1)
struct ModelConfig{
    int dim;
    int hidden_dim;
    int n_layers;
    int q_heads;
    int kv_heads;
    int vocab_size;
    int seq_len;
};
#pragma pack(pop)
//使用c语言风格fopen+fread  申请内存 -> 建立管道 -> 灌入数据
//申请一块内存 (在你的 main 函数中)：
//ModelConfig config={}; 这一句非常关键。它在当前线程的**栈内存（Stack）**上硬生生划分出了 28 个字节的空间（7 个 int，由 #pragma pack 保证紧凑）。
//初始化为 {} 意味着这 28 个字节全被填成了 00000000。这就好比你准备好了一个容积为 28 升的空桶，等待装水。
//建立文件管道 (在你的 read_config_1 中)：
//FILE *file = fopen(filepath,"rb");。当这行代码执行时，你的程序会向操作系统发起系统调用。操作系统会在内核空间找到磁盘上的 stories15M.bin，并为你创建一个文件描述符（管道）。
// FILE* 就是这个管道的控制手柄，"rb" 明确规定了这是一根只传输纯粹二进制流（不转义任何字符）的管道。
//像流水一样切下数据灌入 (核心操作)：
//fread(&config, sizeof(ModelConfig), 1, file);。
//&config：告诉操作系统，水桶的起始物理地址在哪里。
//sizeof(ModelConfig)：定义了我们要切下多大一块数据（28字节）。
//底层发生的事：磁盘驱动器开始旋转/寻址，把二进制流先抽到操作系统的内存缓冲区，
// 然后 CPU 将这 28 个字节的数据，一丝不差地强行覆盖到 config 所在的栈内存上。管你这 28 字节原来是 0 还是垃圾值，全被物理覆盖。

bool read_config_c (const char* filepath,ModelConfig& config){
    if(!std::filesystem::exists(filepath)){
        std::cerr<<"[Error]文件不存在"<<std::filesystem::absolute(filepath)<<'\n';
        return false;
    }
    FILE *file= fopen(filepath,"rb");
    if(file== nullptr){
        std::cerr<<"[Error]文件打不开(C fopen,fread)\n";
        return false;
    }
    int res= fread(&config,sizeof (ModelConfig),1,file);
    fclose(file);
    if(res!=1){
        std::cerr<<"[Error]文件读取不成功 C\n";
        return false;
    }else{
        return true;
    }
}

//现代c++风格
//不推荐---不够清晰
bool read_config_cpp(const char* filepath,ModelConfig& config){
    if(!std::filesystem::exists(filepath)){
        std::cerr<<"[Error]文件不存在"<<std::filesystem::absolute(filepath)<<'\n';
        return false;
    }
    std::ifstream file(filepath,std::ios::binary);
    if(!file.is_open()){
        std::cerr<<"[Error]文件打不开(C++ ifstream)\n";
        return false;
    }
    file.read(reinterpret_cast<char*>(&config),sizeof (ModelConfig));
    if(!file){
        std::cerr << "[Error] C++ ifstream 读取字节流失败\n";
        return false;
    }
    file.close();
    return true;
}

//mmap内存映射,零拷贝
bool read_config_mmap(const char* filepath,ModelConfig& config){
    if(!std::filesystem::exists(filepath)){
        std::cout<<"[Error]文件不存在\n";
        return false;
    }
    int fd=open(filepath,O_RDONLY);
    if(fd==-1){
        std::cerr<<"[Error]open系统调用失败\n";
        return false;
    }
    struct stat sb;
    if(fstat(fd,&sb)==-1){
        close(fd);
        return false;
    }
    size_t file_size=sb.st_size;
    void* mapped_data=mmap(NULL,file_size,PROT_READ,MAP_PRIVATE,fd,0);
    if(mapped_data==MAP_FAILED){
        std::cerr<<"[Error]mmap映射失败\n";
        close(fd);
        return false;
    }
    ModelConfig* mmap_config_ptr=static_cast<ModelConfig*>(mapped_data);
    config=*mmap_config_ptr;
    munmap(mapped_data,file_size);
    close(fd);
    return true;
}


int main(){
    ModelConfig config={};
    std::cout<<"---读取Llama2-15M的模型参数---\n";
    char* filepath="models/stories15M.bin";
    if(read_config_c(filepath,config)){
        std::cout<<"C模式读取文件成功\n";
        std::cout<<"-----模型配置-----\n";
        std::cout<<"特征维度:"<<config.dim<<'\n';
        std::cout<<"隐藏层维度:"<<config.hidden_dim<<'\n';
        std::cout<<"模型层数:"<<config.n_layers<<'\n';
        std::cout<<"Q的头数:"<<config.q_heads<<'\n';
        std::cout<<"KV的头数:"<<config.kv_heads<<'\n';
        std::cout<<"词表大小:"<<config.vocab_size<<'\n';
        std::cout<<"支持的最大句子长度:"<<config.seq_len<<'\n';
    }else{
        std::cerr<<"读取失败,程序终止\n";
    }
    if(read_config_cpp(filepath,config)){
        std::cout<<"C++模式读取文件成功\n";
        std::cout<<"-----模型配置-----\n";
        std::cout<<"特征维度:"<<config.dim<<'\n';
        std::cout<<"隐藏层维度:"<<config.hidden_dim<<'\n';
        std::cout<<"模型层数:"<<config.n_layers<<'\n';
        std::cout<<"Q的头数:"<<config.q_heads<<'\n';
        std::cout<<"KV的头数:"<<config.kv_heads<<'\n';
        std::cout<<"词表大小:"<<config.vocab_size<<'\n';
        std::cout<<"支持的最大句子长度:"<<config.seq_len<<'\n';
    }else{
        std::cerr<<"读取失败,程序终止\n";
    }
    if(read_config_mmap(filepath,config)){
        std::cout<<"mmap模式读取文件成功\n";
        std::cout<<"-----模型配置-----\n";
        std::cout<<"特征维度:"<<config.dim<<'\n';
        std::cout<<"隐藏层维度:"<<config.hidden_dim<<'\n';
        std::cout<<"模型层数:"<<config.n_layers<<'\n';
        std::cout<<"Q的头数:"<<config.q_heads<<'\n';
        std::cout<<"KV的头数:"<<config.kv_heads<<'\n';
        std::cout<<"词表大小:"<<config.vocab_size<<'\n';
        std::cout<<"支持的最大句子长度:"<<config.seq_len<<'\n';
    }else{
        std::cerr<<"读取失败,程序终止\n";
    }

}