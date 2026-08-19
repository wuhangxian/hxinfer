#include "cstring"
#include "cstdlib"
#include "iostream"
#include "memory"
#include "vector"
#include "stdexcept"
#include "numeric"
#include "cmath"
#include "limits"
#include <chrono>
struct ModelConfig{
    int dim;
    int hidden_dim;
    int n_layer;
    int n_head;
    int n_kv_head;
    int vocab_size;
    int seq_len;
};

class Allocator{
public:
    virtual void* allocate(size_t byte_size)=0;
    virtual void release(void* ptr)=0;
    virtual ~Allocator()=default;
};

class CPUAllocator:public Allocator{
public:
    void* allocate(size_t byte_size) override{
        void* ptr= nullptr;
        int ret= posix_memalign(&ptr,64,byte_size);
        if(ret!=0||ptr== nullptr){
            std::cerr<<"内存申请未成功,内存大小不够了"<<'\n';
            throw std::bad_alloc();
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
    void* data_;
public:
    Buffer(size_t byte_size,std::shared_ptr<Allocator> allocator):
            allocator_(allocator),byte_size_(byte_size){
        if(allocator_== nullptr){
            throw std::invalid_argument("Allocator不能为空指针!\n");
        }
        data_=allocator_->allocate(byte_size);
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
    const void* buffer_data_ptr() const{
        return data_;
    }
    void* buffer_data_ptr(){
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
        total_elements_=std::accumulate(shapes_.begin(),shapes_.end(),size_t{1},std::multiplies<>());
        size_t byte_size=total_elements_*sizeof (float );
        buffer_=std::make_shared<Buffer>(byte_size,allocator);
    }
    ~Tensor()=default;
    const std::vector<int>& tensor_shapes() const{
        return shapes_;
    }
    void set_shape(std::vector<int> new_shapes){
        size_t new_total=std::accumulate(new_shapes.begin(),new_shapes.end(),size_t{1},std::multiplies<>());
        size_t max_capacity_elements=buffer_->buffer_byte_size()/sizeof (float );
        if(new_total>max_capacity_elements){
            throw std::runtime_error("Tensor reshape  失败:新形状超出底层申请的物理容量!");
        }
        shapes_=new_shapes;
        total_elements_=new_total;
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
    void tensor_print_data() const{
        int n=shapes_.size();
        const float *p=static_cast<const float *>(tensor_data_ptr());
        if(n==1){
            int len=shapes_[0];
            std::cout<<'[';
            for(int i=0;i<len;i++){
                std::cout<<p[i]<<' ';
            }
            std::cout<<']'<<'\n';
        }
        if(n==2){
            int row=shapes_[0];
            int col=shapes_[1];
            const float *curr=p;
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
            int space=shapes_[0];
            int row=shapes_[1];
            int col=shapes_[2];
            const float *curr=p;
            std::cout<<'['<<'\n';
            for(int i=0;i<space;i++){
                std::cout<<'['<<'\n';
                for(int j=0;j<row;j++){
                    std::cout<<'[';
                    curr=p+i*row*col+j*col;
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

class Layer{
public:
    std::string layer_name_;
    Layer(std::string layer_name):layer_name_(layer_name){}
    virtual void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output)=0;
    virtual ~Layer()=default;
};

class MathOps{
public:
    static void add_tensors(std::shared_ptr<Tensor> input_a,std::shared_ptr<Tensor> input_b,std::shared_ptr<Tensor> output){
        float *a_ptr=static_cast<float *>(input_a->tensor_data_ptr());
        float *b_ptr=static_cast<float *>(input_b->tensor_data_ptr());
        float *out_ptr=static_cast<float *>(output->tensor_data_ptr());
        size_t n=input_a->tensor_total_elements();
        for(size_t i=0;i<n;i++){
            out_ptr[i]=a_ptr[i]+b_ptr[i];
        }
    }
    static void mul_tensors(std::shared_ptr<Tensor> input_a,std::shared_ptr<Tensor> input_b,std::shared_ptr<Tensor> output){
        float *a_ptr=static_cast<float *>(input_a->tensor_data_ptr());
        float *b_ptr=static_cast<float *>(input_b->tensor_data_ptr());
        float *out_ptr=static_cast<float *>(output->tensor_data_ptr());
        size_t n=input_a->tensor_total_elements();
        for(size_t i=0;i<n;i++){
            out_ptr[i]=a_ptr[i]*b_ptr[i];
        }
    }
    static void silu_tensor(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output){
        float *in_ptr=static_cast<float *>(input->tensor_data_ptr());
        float *out_ptr=static_cast<float *>(output->tensor_data_ptr());
        size_t n=input->tensor_total_elements();
        for(size_t i=0;i<n;i++){
            out_ptr[i]=in_ptr[i]/(1+std::exp(-in_ptr[i]));
        }
    }
    static void softmax_tensor(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output){
        float *in_ptr=static_cast<float *>(input->tensor_data_ptr());
        float *out_ptr=static_cast<float *>(output->tensor_data_ptr());
        std::vector<int> in_shapes=input->tensor_shapes();
        int dims=in_shapes[in_shapes.size()-1];
        size_t row=input->tensor_total_elements()/dims;
        for(size_t i=0;i<row;i++){
            float max_val=std::numeric_limits<float>::lowest();
            float *curr_in=in_ptr+i*dims;
            float *curr_out=out_ptr+i*dims;
            float sum=0;
            for(int j=0;j<dims;j++){
                if(curr_in[j]>max_val){
                    max_val=curr_in[j];
                }
            }
            for(int j=0;j<dims;j++){
                curr_out[j]=std::exp(curr_in[j]-max_val);
                sum=sum+curr_out[j];
            }
            float mid=1/sum;
            for(int j=0;j<dims;j++){
                curr_out[j]=curr_out[j]*mid;
            }
        }
    }
    static void rope(std::shared_ptr<Tensor> q,std::shared_ptr<Tensor> k,ModelConfig& config){
        float *q_ptr=static_cast<float *>(q->tensor_data_ptr());
        float *k_ptr=static_cast<float *>(k->tensor_data_ptr());
        int dim=config.dim;
        int n_head=config.n_head;
        int n_dim=dim/n_head;
        std::vector<int> shapes=q->tensor_shapes();
        int seq_len=shapes[0];
        for(int i=0;i<seq_len;i++){
            for(int j=0;j<n_head;j++){
                for(int k=0;k<n_dim;k=k+2){
                    float freq=1.0/std::pow(10000,(float )k/(float )n_dim);
                    float angle=i*freq;
                    float angle_cos=std::cos(angle);
                    float angle_sin=std::sin(angle);
                    int offset=i*dim+j*n_dim+k;
                    float q0=q_ptr[offset];
                    float q1=q_ptr[offset+1];
                    q_ptr[offset]=q0*angle_cos-q1*angle_sin;
                    q_ptr[offset+1]=q0*angle_sin+q1*angle_cos;
                    float k0=k_ptr[offset];
                    float k1=k_ptr[offset+1];
                    k_ptr[offset]=k0*angle_cos-k1*angle_sin;
                    k_ptr[offset+1]=k0*angle_sin+k1*angle_cos;
                }
            }
        }
    }
};

class RMSNormLayer:public Layer{
private:
    std::shared_ptr<Tensor> weights_;
    float eps_;
public:
    RMSNormLayer(std::shared_ptr<Tensor> weight,float eps=1e-5):weights_(weight),eps_(eps), Layer("RMSNorm"){}
    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override{
        float *in_ptr=static_cast<float *>(input->tensor_data_ptr());
        float *out_ptr=static_cast<float *>(output->tensor_data_ptr());
        float *w_ptr=static_cast<float *>(weights_->tensor_data_ptr());
        std::vector<int> in_shapes=input->tensor_shapes();
        int dims=in_shapes[in_shapes.size()-1];
        size_t row=input->tensor_total_elements()/dims;
        for(size_t i=0;i<row;i++){
            float sum=0;
            float *curr_in=in_ptr+i*dims;
            float *curr_out=out_ptr+i*dims;
            for(int j=0;j<dims;j++){
                sum=sum+curr_in[j]*curr_in[j];
            }
            float RMS=std::sqrt(eps_+sum/dims);
            for(int j=0;j<dims;j++){
                curr_out[j]=curr_in[j]/RMS*w_ptr[j];
            }
        }
    }
};

class LinearLayer:public Layer{
private:
    std::shared_ptr<Tensor> weights_;
public:
    LinearLayer(std::shared_ptr<Tensor> weights): Layer("Linear"),weights_(weights){}

    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override{
        float *in_ptr=static_cast<float *>(input->tensor_data_ptr());
        float *out_ptr=static_cast<float *>(output->tensor_data_ptr());
        float *w_ptr=static_cast<float *>(weights_->tensor_data_ptr());
        std::vector<int> in_shapes=input->tensor_shapes();
        std::vector<int> w_shapes=weights_->tensor_shapes();
        int K=in_shapes[in_shapes.size()-1];
        size_t M=input->tensor_total_elements()/K;
        int N=w_shapes[0];
        for(size_t i=0;i<M;i++){
            float* curr_out=out_ptr+i*N;
            float *curr_in=in_ptr+i*K;
            for(int j=0;j<N;j++){
                float *curr_w=w_ptr+j*K;
                float sum=0;
                for(int k=0;k<K;k++){
                    sum=sum+curr_in[k]*curr_w[k];
                }
                curr_out[j]=sum;
            }
        }
    }
};

class SwiGLULayer:public Layer{
private:
    std::shared_ptr<LinearLayer> layer_gate_;
    std::shared_ptr<LinearLayer> layer_up_;
    std::shared_ptr<LinearLayer> layer_down_;
    std::shared_ptr<Tensor> after_silu_tensor_;
    std::shared_ptr<Tensor> after_up_tensor_;
    int hidden_dim_;
    int max_seq_len_;
public:
    SwiGLULayer(std::shared_ptr<LinearLayer> layer_gate,std::shared_ptr<LinearLayer> layer_up,std::shared_ptr<LinearLayer> layer_down,
                const ModelConfig& config,std::shared_ptr<Allocator> allocator):
                layer_gate_(layer_gate),layer_up_(layer_up),layer_down_(layer_down),
                Layer("SwiGLU"),hidden_dim_(config.hidden_dim),max_seq_len_(config.seq_len){
        std::vector<int> max_shape={max_seq_len_,hidden_dim_};
        after_silu_tensor_=std::make_shared<Tensor>(max_shape,allocator);
        after_up_tensor_=std::make_shared<Tensor>(max_shape,allocator);
    }
    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override{
        std::vector<int> in_shapes=input->tensor_shapes();
        int curr_seq_len=in_shapes[0];
        std::vector<int> curr_hidden_shape={curr_seq_len,hidden_dim_};
        after_silu_tensor_->set_shape(curr_hidden_shape);
        after_up_tensor_->set_shape(curr_hidden_shape);
        layer_gate_->forward(input,after_silu_tensor_);
        MathOps::silu_tensor(after_silu_tensor_,after_silu_tensor_);
        layer_up_->forward(input, after_up_tensor_);
        MathOps::mul_tensors(after_silu_tensor_,after_up_tensor_,after_silu_tensor_);
        layer_down_->forward(after_silu_tensor_,output);
    }
};

class LlamaAttentionLayer:public Layer{
private:
    std::shared_ptr<LinearLayer> wq_;
    std::shared_ptr<LinearLayer> wk_;
    std::shared_ptr<LinearLayer> wv_;
    std::shared_ptr<LinearLayer> wo_;

    std::shared_ptr<Tensor> after_q_tensor_;
    std::shared_ptr<Tensor> after_k_tensor_;
    std::shared_ptr<Tensor> after_v_tensor_;
    std::shared_ptr<Tensor> after_qkt_tensor_;
    std::shared_ptr<Tensor> after_qktv_tensor_;

    ModelConfig config_;
public:
    LlamaAttentionLayer(std::shared_ptr<LinearLayer> wq,std::shared_ptr<LinearLayer> wk,
                   std::shared_ptr<LinearLayer> wv,std::shared_ptr<LinearLayer> wo,
                   ModelConfig& config,std::shared_ptr<Allocator> allocator):wq_(wq),wk_(wk),wv_(wv),wo_(wo),
                                                                             config_(config), Layer("LlamaAttention"){
        std::vector<int> max_qkv_shape={config_.seq_len,config_.dim};
        after_q_tensor_=std::make_shared<Tensor>(max_qkv_shape,allocator);
        after_k_tensor_=std::make_shared<Tensor>(max_qkv_shape,allocator);
        after_v_tensor_=std::make_shared<Tensor>(max_qkv_shape,allocator);
        after_qktv_tensor_=std::make_shared<Tensor>(max_qkv_shape,allocator);
        std::vector<int> max_score_shape={config_.n_head,config_.seq_len,config_.seq_len};
        after_qkt_tensor_=std::make_shared<Tensor>(max_score_shape,allocator);
    }
    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override{
        std::vector<int> in_shapes=input->tensor_shapes();
        int curr_seq_len=in_shapes[0];
        std::vector<int> curr_qkv_shape={curr_seq_len,config_.dim};
        after_q_tensor_->set_shape(curr_qkv_shape);
        after_k_tensor_->set_shape(curr_qkv_shape);
        after_v_tensor_->set_shape(curr_qkv_shape);
        after_qktv_tensor_->set_shape(curr_qkv_shape);
        std::vector<int> curr_socre_shape={config_.n_head,curr_seq_len,curr_seq_len};
        after_qkt_tensor_->set_shape(curr_socre_shape);
        wq_->forward(input,after_q_tensor_);
        wk_->forward(input,after_k_tensor_);
        wv_->forward(input,after_v_tensor_);
        MathOps::rope(after_q_tensor_,after_k_tensor_,config_);
        float *q_ptr=static_cast<float *>(after_q_tensor_->tensor_data_ptr());
        float *k_ptr=static_cast<float *>(after_k_tensor_->tensor_data_ptr());
        float *v_ptr=static_cast<float *>(after_v_tensor_->tensor_data_ptr());
        float *qkt_ptr=static_cast<float *>(after_qkt_tensor_->tensor_data_ptr());
        float *qktv_ptr=static_cast<float *>(after_qktv_tensor_->tensor_data_ptr());
        int n_head=config_.n_head;
        int dim=config_.dim;
        int head_dim=dim/n_head;
        float scale=1.0/std::sqrt(static_cast<float>(head_dim));
        for(int i=0;i<n_head;i++){
            for(int j=0;j<curr_seq_len;j++){
                for(int k=0;k<curr_seq_len;k++){
                    int qkt_index=i*curr_seq_len*curr_seq_len+j*curr_seq_len+k;
                    if(k>j){
                        qkt_ptr[qkt_index]=-1e9f;
                    }else{
                        float sum=0;
                        float *curr_q=q_ptr+j*dim+i*head_dim;
                        float *curr_k=k_ptr+k*dim+i*head_dim;
                        for(int sc=0;sc<head_dim;sc++){
                            sum=sum+curr_q[sc]*curr_k[sc];
                        }
                        qkt_ptr[qkt_index]=sum*scale;
                    }

                }
            }
        }
        MathOps::softmax_tensor(after_qkt_tensor_,after_qkt_tensor_);
        for(int i=0;i<n_head;i++){
            for(int j=0;j<curr_seq_len;j++){
                for(int k=0;k<head_dim;k++){
                    float sum=0;
                    int out_offset=j*dim+i*head_dim+k;
                    for(int sc=0;sc<=j;sc++){
                        int qkt_index=i*curr_seq_len*curr_seq_len+j*curr_seq_len+sc;
                        int v_index=sc*dim+i*head_dim+k;
                        sum=sum+qkt_ptr[qkt_index]*v_ptr[v_index];
                    }
                    qktv_ptr[out_offset]=sum;
                }
            }
        }
        wo_->forward(after_qktv_tensor_,output);
    }
};

class TransformerBlockLayer : public Layer {
private:
    // 1. 核心大组件 (图纸里的四大金刚)
    std::shared_ptr<RMSNormLayer> attn_norm_;
    std::shared_ptr<LlamaAttentionLayer> attention_;
    std::shared_ptr<RMSNormLayer> ffn_norm_;
    std::shared_ptr<SwiGLULayer> ffn_;

    // 2. 内部连线用的“草稿纸” (临时 Buffer)
    std::shared_ptr<Tensor> norm_buf_;     // 用来存 RMSNorm 的结果
    std::shared_ptr<Tensor> attn_out_buf_; // 用来存 Attention 算出来的增量
    std::shared_ptr<Tensor> ffn_out_buf_;  // 用来存 SwiGLU 算出来的增量

    int dim_;

public:
    // 构造函数：把外部已经准备好的四个组件注进来
    TransformerBlockLayer(std::shared_ptr<RMSNormLayer> attn_norm,
                          std::shared_ptr<LlamaAttentionLayer> attention,
                          std::shared_ptr<RMSNormLayer> ffn_norm,
                          std::shared_ptr<SwiGLULayer> ffn,
                          ModelConfig& config,
                          std::shared_ptr<Allocator> alloc)
            : Layer("TransformerBlock"),
              attn_norm_(attn_norm), attention_(attention),
              ffn_norm_(ffn_norm), ffn_(ffn),
              dim_(config.dim) {

        // 预分配草稿纸：所有的中间结果，最大也就是 [max_seq_len, dim]
        std::vector<int> max_shape = {config.seq_len, config.dim};
        norm_buf_ = std::make_shared<Tensor>(max_shape, alloc);
        attn_out_buf_ = std::make_shared<Tensor>(max_shape, alloc);
        ffn_out_buf_ = std::make_shared<Tensor>(max_shape, alloc);
    }

    void forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output) override {
        // 1. 获取当前真实的句子长度
        std::vector<int> in_shapes = input->tensor_shapes();
        int curr_seq_len = in_shapes[0];
        std::vector<int> curr_shape = {curr_seq_len, dim_};

        // 2. 动态调整草稿纸大小
        norm_buf_->set_shape(curr_shape);
        attn_out_buf_->set_shape(curr_shape);
        ffn_out_buf_->set_shape(curr_shape);

        output->set_shape(curr_shape);
        // ==========================================
        // 上半场：Attention 模块与第一次残差连接
        // ==========================================

        // 步骤 A: 对输入进行归一化 -> 存入 norm_buf_
        attn_norm_->forward(input, norm_buf_);

        // 步骤 B: 进行注意力计算 -> 存入 attn_out_buf_
        attention_->forward(norm_buf_, attn_out_buf_);

        // 步骤 C: 残差连接 (Output = Input + Attention输出)
        // 注意：这里我们直接把结果写到了最终的 output 里面
        MathOps::add_tensors(input, attn_out_buf_, output);


        // ==========================================
        // 下半场：FFN 前馈网络模块与第二次残差连接
        // ==========================================

        // 步骤 D: 对刚才的 output 再次归一化 -> 覆盖存入 norm_buf_ (内存复用！)
        ffn_norm_->forward(output, norm_buf_);

        // 步骤 E: 进行 SwiGLU 计算 -> 存入 ffn_out_buf_
        ffn_->forward(norm_buf_, ffn_out_buf_);

        // 步骤 F: 第二次残差连接 (Output = Output + FFN输出)
        // 在原地把自己加上去，大功告成！
        MathOps::add_tensors(output, ffn_out_buf_, output);
    }
};

class EmbeddingLayer : public Layer {
private:
    std::shared_ptr<Tensor> weights_; // 这是一个极其庞大的矩阵：[vocab_size, dim]
    int dim_;

public:
    EmbeddingLayer(std::shared_ptr<Tensor> weights, ModelConfig& config)
            : Layer("Embedding"), weights_(weights), dim_(config.dim) {}

    // 重载一个特殊的 forward，因为输入不再是浮点数 Tensor，而是整数 Token 数组
    void forward_tokens(const std::vector<int>& tokens, std::shared_ptr<Tensor> output) {
        float* w_ptr = static_cast<float*>(weights_->tensor_data_ptr());
        float* out_ptr = static_cast<float*>(output->tensor_data_ptr());

        int seq_len = tokens.size();

        // 动态调整输出 Tensor 的形状
        output->set_shape({seq_len, dim_});

        // 查字典并拷贝！
        for (int i = 0; i < seq_len; i++) {
            int token_id = tokens[i];

            // 严谨的安全校验：防止查字典越界导致段错误
            // Llama2 的 vocab_size 通常是 32000
            if (token_id < 0 || token_id >= weights_->tensor_shapes()[0]) {
                throw std::runtime_error("Token ID 越界！");
            }

            // 🌟 极致性能优化：不要用 for 循环一个一个 float 赋值！
            // 直接用底层的 memcpy (内存拷贝)，把字典里的一整行直接按字节搬运过去
            std::memcpy(out_ptr + i * dim_,
                        w_ptr + token_id * dim_,
                        dim_ * sizeof(float));
        }
    }

    // 继承自 Layer 的纯虚函数，这里置空即可，我们主要用 forward_tokens
    void forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output) override {}
};

class LlamaModel {
private:
    std::shared_ptr<EmbeddingLayer> embedding_;
    // Llama 有多层 Block，我们用 std::vector 装起来
    std::vector<std::shared_ptr<TransformerBlockLayer>> blocks_;
    std::shared_ptr<RMSNormLayer> final_norm_;
    std::shared_ptr<LinearLayer> lm_head_; // 最后的输出预测层

    // 整个大模型只需要两张草稿纸来回倒腾数据 (Ping-Pong Buffer)
    std::shared_ptr<Tensor> hidden_state_1_;
    std::shared_ptr<Tensor> hidden_state_2_;

    ModelConfig config_;

public:
    // 构造函数：注入所有已经装载好权重的层！
    LlamaModel(std::shared_ptr<EmbeddingLayer> embedding,
               std::vector<std::shared_ptr<TransformerBlockLayer>> blocks,
               std::shared_ptr<RMSNormLayer> final_norm,
               std::shared_ptr<LinearLayer> lm_head,
               ModelConfig& config,
               std::shared_ptr<Allocator> alloc)
            : embedding_(embedding), blocks_(blocks),
              final_norm_(final_norm), lm_head_(lm_head), config_(config) {

        // 预分配终极草稿纸：最大容量 [max_seq_len, dim]
        std::vector<int> max_shape = {config_.seq_len, config_.dim};
        hidden_state_1_ = std::make_shared<Tensor>(max_shape, alloc);
        hidden_state_2_ = std::make_shared<Tensor>(max_shape, alloc);
    }

    // 🌟 整个大模型的前向传播主控台
    // 输入：一串 Token IDs
    // 输出：模型算出的 Logits 打分矩阵
    void forward(const std::vector<int>& tokens, std::shared_ptr<Tensor> logits_output) {

        // 1. 过 Embedding 层：查字典，把结果写进草稿纸 1
        embedding_->forward_tokens(tokens, hidden_state_1_);

        // 2. 穿越无尽的 Transformer Block！
        // 核心技巧：Ping-Pong 轮流写，省去所有的中间内存申请
        std::shared_ptr<Tensor> curr_in = hidden_state_1_;
        std::shared_ptr<Tensor> curr_out = hidden_state_2_;

        for (int i = 0; i < blocks_.size(); i++) {
            // 跑一层 Block
            blocks_[i]->forward(curr_in, curr_out);

            // 交换指针：这一层的输出，变成下一层的输入！
            std::swap(curr_in, curr_out);
        }

        // 循环结束后，最终的特征藏在 curr_in 里面！
        curr_out->set_shape(curr_in->tensor_shapes());
        // 3. 全局最终归一化 (Final RMSNorm)
        // 从 curr_in 读，写回 curr_out
        final_norm_->forward(curr_in, curr_out);

        // 4. LM Head 输出投影 (从隐藏维度 dim 映射回庞大的词汇表维度 vocab_size)
        // 算出每个词的概率打分，写入用户提供的 logits_output 里面
        lm_head_->forward(curr_out, logits_output);
    }
};

// 词表翻译官：负责把 Token ID 变成人类文字
class Tokenizer {
private:
    std::vector<std::string> vocab_; // 存 32000 个词的数组

public:
    Tokenizer(const char* filepath, int vocab_size) {
        FILE* file = fopen(filepath, "rb");
        if (!file) {
            throw std::runtime_error("找不到 tokenizer.bin 文件！");
        }

        // 1. 读取最长的词有多长 (用来准备草稿纸)
        int max_token_length;
        fread(&max_token_length, sizeof(int), 1, file);

        vocab_.resize(vocab_size);
        std::vector<char> word_buf(max_token_length + 1); // 字符串读取草稿纸

        // 2. 循环读取 32000 个词
        for (int i = 0; i < vocab_size; i++) {
            float score; // 词的得分 (我们推理时用不到，但也得读出来跳过)
            fread(&score, sizeof(float), 1, file);

            int len;     // 这个词有几个字母
            fread(&len, sizeof(int), 1, file);

            fread(word_buf.data(), 1, len, file); // 读字母
            word_buf[len] = '\0'; // 加上 C++ 字符串的结束符

            vocab_[i] = std::string(word_buf.data());
        }
        fclose(file);
    }

    // 给一个 ID，还你一个单词
    std::string decode(int token_id) {
        if (token_id >= 0 && token_id < vocab_.size()) {
            return vocab_[token_id];
        }
        return "";
    }
};
// 放在 main 函数的上面
void load_tensor_from_file(FILE* file, std::shared_ptr<Tensor> tensor) {
    size_t elements = tensor->tensor_total_elements();
    size_t read_items = fread(tensor->tensor_data_ptr(), sizeof(float), elements, file);
    if (read_items != elements) {
        throw std::runtime_error("读取权重失败！文件可能过早结束或损坏。");
    }
}

// 🌟 核心封装：把又长又臭的组装逻辑全部打包到这里面！
std::shared_ptr<LlamaModel> build_llama_model(const std::string& model_path, ModelConfig& config, std::shared_ptr<Allocator> alloc) {
    FILE* file = fopen(model_path.c_str(), "rb");
    if (!file) throw std::runtime_error("找不到权重文件！");

    fread(&config, sizeof(int), 7, file);
    std::cout << "模型配置读取成功，开始装配高达..." << '\n';

    auto embed_weight = std::make_shared<Tensor>(std::vector<int>{config.vocab_size, config.dim}, alloc);
    load_tensor_from_file(file, embed_weight);
    auto embedding_layer = std::make_shared<EmbeddingLayer>(embed_weight, config);

    std::vector<std::shared_ptr<Tensor>> attn_norm_w(config.n_layer);
    for (int i=0; i<config.n_layer; i++) { attn_norm_w[i] = std::make_shared<Tensor>(std::vector<int>{config.dim}, alloc); load_tensor_from_file(file, attn_norm_w[i]); }

    std::vector<std::shared_ptr<Tensor>> wq_w(config.n_layer), wk_w(config.n_layer), wv_w(config.n_layer), wo_w(config.n_layer);
    for (int i=0; i<config.n_layer; i++) { wq_w[i] = std::make_shared<Tensor>(std::vector<int>{config.dim, config.dim}, alloc); load_tensor_from_file(file, wq_w[i]); }
    for (int i=0; i<config.n_layer; i++) { wk_w[i] = std::make_shared<Tensor>(std::vector<int>{config.dim, config.dim}, alloc); load_tensor_from_file(file, wk_w[i]); }
    for (int i=0; i<config.n_layer; i++) { wv_w[i] = std::make_shared<Tensor>(std::vector<int>{config.dim, config.dim}, alloc); load_tensor_from_file(file, wv_w[i]); }
    for (int i=0; i<config.n_layer; i++) { wo_w[i] = std::make_shared<Tensor>(std::vector<int>{config.dim, config.dim}, alloc); load_tensor_from_file(file, wo_w[i]); }

    std::vector<std::shared_ptr<Tensor>> ffn_norm_w(config.n_layer);
    for (int i=0; i<config.n_layer; i++) { ffn_norm_w[i] = std::make_shared<Tensor>(std::vector<int>{config.dim}, alloc); load_tensor_from_file(file, ffn_norm_w[i]); }

    std::vector<std::shared_ptr<Tensor>> w1_w(config.n_layer), w2_w(config.n_layer), w3_w(config.n_layer);
    for (int i=0; i<config.n_layer; i++) { w1_w[i] = std::make_shared<Tensor>(std::vector<int>{config.hidden_dim, config.dim}, alloc); load_tensor_from_file(file, w1_w[i]); }
    for (int i=0; i<config.n_layer; i++) { w2_w[i] = std::make_shared<Tensor>(std::vector<int>{config.dim, config.hidden_dim}, alloc); load_tensor_from_file(file, w2_w[i]); }
    for (int i=0; i<config.n_layer; i++) { w3_w[i] = std::make_shared<Tensor>(std::vector<int>{config.hidden_dim, config.dim}, alloc); load_tensor_from_file(file, w3_w[i]); }

    auto final_norm_w = std::make_shared<Tensor>(std::vector<int>{config.dim}, alloc);
    load_tensor_from_file(file, final_norm_w);
    auto final_norm_layer = std::make_shared<RMSNormLayer>(final_norm_w);

    int head_size = config.dim / config.n_head;
    fseek(file, config.seq_len * (head_size / 2) * sizeof(float) * 2, SEEK_CUR);

    auto lm_head_w = std::make_shared<Tensor>(std::vector<int>{config.vocab_size, config.dim}, alloc);
    if (fgetc(file) == EOF) lm_head_w = embed_weight;
    else { fseek(file, -1, SEEK_CUR); load_tensor_from_file(file, lm_head_w); }
    auto lm_head_layer = std::make_shared<LinearLayer>(lm_head_w);
    fclose(file);

    std::vector<std::shared_ptr<TransformerBlockLayer>> blocks;
    for (int i = 0; i < config.n_layer; i++) {
        auto attn = std::make_shared<LlamaAttentionLayer>(std::make_shared<LinearLayer>(wq_w[i]), std::make_shared<LinearLayer>(wk_w[i]), std::make_shared<LinearLayer>(wv_w[i]), std::make_shared<LinearLayer>(wo_w[i]), config, alloc);
        auto ffn = std::make_shared<SwiGLULayer>(std::make_shared<LinearLayer>(w1_w[i]), std::make_shared<LinearLayer>(w3_w[i]), std::make_shared<LinearLayer>(w2_w[i]), config, alloc);
        blocks.push_back(std::make_shared<TransformerBlockLayer>(std::make_shared<RMSNormLayer>(attn_norm_w[i]), attn, std::make_shared<RMSNormLayer>(ffn_norm_w[i]), ffn, config, alloc));
    }
    return std::make_shared<LlamaModel>(embedding_layer, blocks, final_norm_layer, lm_head_layer, config, alloc);
}


// 🌟 极简清爽版 main 函数 (带极简测速)
int main() {
    std::cout << "========== 正在启动极简 C++ LLaMA 推理引擎 ==========" << '\n';

    ModelConfig config;
    std::shared_ptr<Allocator> alloc = std::make_shared<CPUAllocator>();

    auto llama = build_llama_model("models/stories15M.bin", config, alloc);
    Tokenizer tokenizer("models/tokenizer.bin", config.vocab_size);
    std::cout << "========== 组装完毕，开始生成童话故事！ ==========\n\n";

    auto logits_output = std::make_shared<Tensor>(std::vector<int>{config.seq_len, config.vocab_size}, alloc);
    int next_token = 1;
    std::vector<int> prompt_tokens = {next_token};
    int max_generate_steps = 100;

    // 🕒 记录总时间
    auto total_start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < max_generate_steps; step++) {
        // 🌟 记录单个 Token 推理的开始时间
        auto start_step = std::chrono::high_resolution_clock::now();

        // 极其耗时的全量计算 (相当于你代码里的 model.forward)
        llama->forward(prompt_tokens, logits_output);

        // 找最后一步的输出结果
        float* logits_ptr = static_cast<float*>(logits_output->tensor_data_ptr());
        float* last_token_logits = logits_ptr + (prompt_tokens.size() - 1) * config.vocab_size;

        // 贪心采样找出 next_token
        int best_token = 0;
        float max_val = last_token_logits[0];
        for (int i = 1; i < config.vocab_size; i++) {
            if (last_token_logits[i] > max_val) {
                max_val = last_token_logits[i];
                best_token = i;
            }
        }

        std::string word = tokenizer.decode(best_token);

        // 🌟 记录单个 Token 推理的结束时间，并计算瞬间速度
        auto end_step = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> step_duration = end_step - start_step;
        double step_tok_per_sec = 1.0 / step_duration.count();

        // 打印生成的词，并在后面带上它消耗的瞬间速度
        if (word == "<0x0A>") {
            std::cout << "\n";
        } else {
            // 1. 先用 cout 打印普通的单词
            std::cout << word;
            // 2. 核心魔法：用 C 语言底层的 printf 直接控制保留 2 位小数 (%.2f)
            // 完美避开 <iomanip> 库，既保持了灰色特效，又极其优雅！
            std::printf("\033[90m[%.2f tok/s]\033[0m", step_tok_per_sec);
        }
        std::fflush(stdout); // 强制立刻刷到屏幕上

        prompt_tokens.push_back(best_token);
    }

    // 🕒 记录总结束时间
    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = total_end - total_start;

    std::cout << "\n\n==================================================\n";
    std::cout << "生成结束！总耗时: " << total_duration.count() << " 秒\n";
    std::cout << "平均速度: " << (int)(max_generate_steps / total_duration.count()) << " tokens/s\n";
    std::cout << "==================================================\n";

    return 0;
}
