#include "cstdio"
#include "iostream"
#include "filesystem"

#pragma pack(push,1)
struct ModelConfig{
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
};
#pragma pack(pop)

bool read_config_c_style(const char* filepath,ModelConfig& config){
    if(!std::filesystem::exists(filepath)){
        std::cerr<<"[Error]文件不存在"<<std::filesystem::absolute(filepath)<<'\n';
        return false;
    }
    FILE* file=fopen(filepath,"rb");
    if(file== nullptr){
        std::cerr<<"[Error]无法打开文件:"<<filepath<<'\n';
        return false;
    }
    size_t read_items= fread(&config,sizeof (ModelConfig),1,file);
    fclose(file);
    if(read_items==1){
        std::cout<<"[Success]成功读取配置\n";
        return true;
    }else{
        std::cerr<<"[Error]未成功读取到配置\n";
        return false;
    }
}

int main(){
    char* filepath="models/stories15M.bin";
    ModelConfig config={0};
    bool flag=read_config_c_style(filepath,config);
    if(flag){
        std::cout<<"-----模型配置-----\n";
        std::cout<<"特征维度:"<<config.dim<<'\n';
        std::cout<<"隐藏层维度:"<<config.hidden_dim<<'\n';
        std::cout<<"模型层数:"<<config.n_layers<<'\n';
        std::cout<<"Q的头数:"<<config.n_heads<<'\n';
        std::cout<<"KV的头数:"<<config.n_kv_heads<<'\n';
        std::cout<<"词表大小:"<<config.vocab_size<<'\n';
        std::cout<<"支持的最大句子长度:"<<config.seq_len<<'\n';
    }else{
        std::cerr<<"读取失败,程序终止\n";
    }

}