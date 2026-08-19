#include "cstring"
#include "cstdlib"
#include "iostream"
#include "memory"
#include "vector"
#include "numeric"
#include "cmath"
struct Config{
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
};
class Allocator{
public:
    virtual void *allocate(size_t byte_size)=0;
    virtual void release(void* ptr)=0;
    virtual ~Allocator(){}
};

class CPUAllocator:public Allocator{
public:
    void *allocate(size_t byte_size)override{
        void *ptr= nullptr;
        int ret= posix_memalign(&ptr,64,byte_size);
        if(ret!=0||ptr== nullptr){
            std::cerr<<"[Fatal]内存大小不够,无法申请到内存\n";
            return nullptr;
        }
        memset(ptr,0,byte_size);
        return ptr;
    }
    void release(void *ptr) override{
        if(ptr!= nullptr){
            free(ptr);
        }
    }
};

class Buffer{
private:
    std::shared_ptr<Allocator> allocator_;
    size_t byte_size_;
    void *data_;
public:
    Buffer(size_t byte_size,std::shared_ptr<Allocator> allocator):byte_size_(byte_size),allocator_(allocator){
        data_=allocator_->allocate(byte_size_);
    }
    ~Buffer(){
        if(data_!= nullptr){
            allocator_->release(data_);
            data_= nullptr;
        }
    }
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
    size_t buffer_byte_size() const{
        return byte_size_;
    }
    const void *buffer_data_ptr() const{
        return data_;
    }
    void *buffer_data_ptr(){
        return data_;
    }
};

class Tensor{
private:
    std::vector<int> shapes_;
    std::shared_ptr<Buffer> buffer_;
    size_t total_elements_;
public:
    Tensor(std::vector<int> shapes,std::shared_ptr<Allocator> allocator):
            shapes_(shapes){
        total_elements_=std::accumulate(shapes_.begin(),shapes_.end(),
                                        size_t{1},std::multiplies<>());
        size_t byte_size=total_elements_*sizeof (float );
        buffer_=std::make_shared<Buffer>(byte_size,allocator);
    }
    ~Tensor(){}
    const std::vector<int>& tensor_shapes() const{
        return shapes_;
    }
    size_t tensor_total_elements() const{
        return total_elements_;
    }
    const void* tensor_data_ptr() const{
        return buffer_->buffer_data_ptr();
    }
    void* tensor_data_ptr(){
        return buffer_->buffer_data_ptr();
    }
    void tensor_fill_num(float num){
        float *p=(float *)tensor_data_ptr();
        for(size_t i=0;i<total_elements_;i++){
            p[i]=num;
        }
    }
    void tensor_print_data(){
        int n=shapes_.size();
        float *p=(float *)tensor_data_ptr();
        if(n==1){
            int len=shapes_[0];
            std::cout<<'[';
            for(int i=0;i<len;i++){
                std::cout<<p[i]<<' ';
            }
            std::cout<<']'<<'\n';
        }
        if(n==2){
            float *curr=p;
            int row=shapes_[0];
            int col=shapes_[1];
            std::cout<<'['<<'\n';
            for(int i=0;i<row;i++){
                curr=p+i*col;
                std::cout<<'[';
                for(int j=0;j<col;j++){
                    std::cout<<curr[j]<<' ';
                }
                std::cout<<']'<<'\n';
            }
            std::cout<<']'<<'\n';
        }
        if(n==3){
            float *curr=p;
            int space=shapes_[0];
            int row=shapes_[1];
            int col=shapes_[2];
            std::cout<<'['<<'\n';
            for(int i=0;i<space;i++){
                std::cout<<'['<<'\n';
                for(int j=0;j<row;j++){
                    curr=p+i*row*col+j*col;
                    std::cout<<'[';
                    for(int k=0;k<col;k++){
                        std::cout<<curr[k]<<' ';
                    }
                    std::cout<<']'<<'\n';
                }
                std::cout<<']'<<'\n';
            }
            std::cout<<']'<<'\n';
        }
    }
};

struct LlamaWeights {
    std::shared_ptr<Tensor> token_embedding_table;

    std::shared_ptr<Tensor> rms_att_weight;
    std::shared_ptr<Tensor> wq;
    std::shared_ptr<Tensor> wk;
    std::shared_ptr<Tensor> wv;
    std::shared_ptr<Tensor> wo;

    std::shared_ptr<Tensor> rms_ffn_weight;
    std::shared_ptr<Tensor> w1;
    std::shared_ptr<Tensor> w2;
    std::shared_ptr<Tensor> w3;

    std::shared_ptr<Tensor> rms_final_weight;
    // LM head 的权重（有时候和 embedding 是共享的，我们这里先假设独立读出或者共享）
    std::shared_ptr<Tensor> wcls;
};
// 🌟 新增：暴力灌装机
void load_weights(const char* checkpoint_path, Config* config, LlamaWeights* weights, std::shared_ptr<Allocator> alloc) {
    FILE* file = fopen(checkpoint_path, "rb");
    if (!file) {
        std::cerr << "[Fatal] 找不到模型文件!\n";
        exit(EXIT_FAILURE);
    }

    // 1. 读配置
    fread(config, sizeof(Config), 1, file);

    int dim = config->dim;
    int hidden_dim = config->hidden_dim;
    int layers = config->n_layers;
    int vocab_size = config->vocab_size;

    std::cout << ">>> 开始向物理内存灌注模型权重 (约 60MB)..." << '\n';

    // 2. 动态向系统申请连续内存
    weights->token_embedding_table = std::make_shared<Tensor>(std::vector<int>{vocab_size, dim}, alloc);
    // 注意：这里的形状包含所有的层 (layers)
    weights->rms_att_weight = std::make_shared<Tensor>(std::vector<int>{layers, dim}, alloc);
    weights->wq = std::make_shared<Tensor>(std::vector<int>{layers, dim, dim}, alloc);
    weights->wk = std::make_shared<Tensor>(std::vector<int>{layers, dim, dim}, alloc);
    weights->wv = std::make_shared<Tensor>(std::vector<int>{layers, dim, dim}, alloc);
    weights->wo = std::make_shared<Tensor>(std::vector<int>{layers, dim, dim}, alloc);
    weights->rms_ffn_weight = std::make_shared<Tensor>(std::vector<int>{layers, dim}, alloc);
    weights->w1 = std::make_shared<Tensor>(std::vector<int>{layers, hidden_dim, dim}, alloc);
    weights->w2 = std::make_shared<Tensor>(std::vector<int>{layers, dim, hidden_dim}, alloc);
    weights->w3 = std::make_shared<Tensor>(std::vector<int>{layers, hidden_dim, dim}, alloc);
    weights->rms_final_weight = std::make_shared<Tensor>(std::vector<int>{dim}, alloc);
    weights->wcls = std::make_shared<Tensor>(std::vector<int>{vocab_size, dim}, alloc);

    // 3. 按极其严格的顺序进行连续 fread 暴力读取！
    // 工业界的 C 指针美学：直接强转 tensor_data_ptr 然后写入！
    fread(weights->token_embedding_table->tensor_data_ptr(), sizeof(float), vocab_size * dim, file);
    fread(weights->rms_att_weight->tensor_data_ptr(), sizeof(float), layers * dim, file);
    fread(weights->wq->tensor_data_ptr(), sizeof(float), layers * dim * dim, file);
    fread(weights->wk->tensor_data_ptr(), sizeof(float), layers * dim * dim, file);
    fread(weights->wv->tensor_data_ptr(), sizeof(float), layers * dim * dim, file);
    fread(weights->wo->tensor_data_ptr(), sizeof(float), layers * dim * dim, file);
    fread(weights->rms_ffn_weight->tensor_data_ptr(), sizeof(float), layers * dim, file);
    fread(weights->w1->tensor_data_ptr(), sizeof(float), layers * hidden_dim * dim, file);
    fread(weights->w2->tensor_data_ptr(), sizeof(float), layers * dim * hidden_dim, file);
    fread(weights->w3->tensor_data_ptr(), sizeof(float), layers * hidden_dim * dim, file);
    fread(weights->rms_final_weight->tensor_data_ptr(), sizeof(float), dim, file);

    // 跳过一段不需要的头部数据 (freq_cis_real 和 freq_cis_imag, 这是旧版的遗留，我们自己算 RoPE)
    int head_size = dim / config->n_heads;
    int seq_len = config->seq_len;
    fseek(file, seq_len * (head_size / 2) * sizeof(float) * 2, SEEK_CUR);

    // 读最后也是最大的一个分类器权重 (如果文件到底了没读到，说明它和 embedding 是共享内存的)
    size_t wcls_read = fread(weights->wcls->tensor_data_ptr(), sizeof(float), vocab_size * dim, file);
    if (wcls_read == 0) {
        std::cout << "[提示] 该模型 wcls 与 embedding 共享权重，进行复用。" << '\n';
        weights->wcls = weights->token_embedding_table;
    }

    fclose(file);
    std::cout << ">>> 恭喜！所有真实权重已成功装载完毕！" << '\n';
}
int main(){
    std::cout << "--- 启动 Llama 权重装载系统 ---" << '\n';
    const char* checkpoint_path="models/stories15M.bin";
    Config config;
    std::shared_ptr<Allocator> alloc = std::make_shared<CPUAllocator>();
    LlamaWeights weights;

    // 一键调用，完成所有解析和加载
    load_weights(checkpoint_path, &config, &weights, alloc);

    std::cout << ">>> 模型基因密码读取成功！结构如下：" << '\n';
    std::cout << "  dim (维度)        = " << config.dim << '\n';
    std::cout << "  hidden_dim        = " << config.hidden_dim << '\n';
    std::cout << "  n_layers (层数)   = " << config.n_layers << '\n';
    std::cout << "  n_heads           = " << config.n_heads << '\n';
    std::cout << "  n_kv_heads        = " << config.n_kv_heads << '\n';
    std::cout << "  vocab_size(词表)  = " << config.vocab_size << '\n';
    std::cout << "  seq_len(上下文)   = " << config.seq_len << '\n';

    // 抽查一下！打印 Embedding 表的第 0 个词的前 5 个维度的浮点数
    // 如果不是乱码，说明加载完美成功！
    float* emb_ptr = (float*)weights.token_embedding_table->tensor_data_ptr();
    std::cout << "\n抽查 Embedding[0] 的前 5 个数值 (应该是正常的浮点数如 0.0123...): " << '\n';
    for (int i = 0; i < 5; i++) {
        std::cout << emb_ptr[i] << " ";
    }
    std::cout << '\n';

    return 0;
}