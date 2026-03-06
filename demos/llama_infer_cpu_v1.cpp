#include "cstring"
#include "cstdlib"
#include "stdexcept"
#include "iostream"
#include "memory"
#include "vector"
#include "numeric"
#include "cmath"
#include "chrono"
struct ModelConfig{
    int dims;
    int hidden_dims;
    int layers;
    int heads;
    int kv_heads;
    int vocab_sizes;
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
            std::cerr<<"内存不够了无法被申请\n";
            throw std::bad_alloc();
        }
        memset(ptr,0,byte_size);
        return ptr;
    }
    void release(void* ptr) override{
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
    Buffer(size_t byte_size,std::shared_ptr<Allocator> allocator):allocator_(allocator),byte_size_(byte_size){
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
    size_t max_byte_size_;
public:
    Tensor(std::vector<int> shapes,std::shared_ptr<Allocator> allocator):
                shapes_(shapes){
        total_elements_=std::accumulate(shapes_.begin(),shapes_.end(),size_t{1},std::multiplies<>());
        max_byte_size_=total_elements_*sizeof (float );
        buffer_=std::make_shared<Buffer>(max_byte_size_,allocator);
    }
    ~Tensor()=default;
    const std::vector<int>& tensor_shapes() const{
        return shapes_;
    }
    void tensor_reshapes(std::vector<int> shapes){
        size_t now_total_elements=std::accumulate(shapes.begin(),shapes.end(),size_t{1},std::multiplies<>());
        if(max_byte_size_>=now_total_elements*sizeof (float )){
            shapes_=shapes;
            total_elements_=now_total_elements;
        }else{
            throw std::runtime_error("Tensor内存大小不够,不能进行reshape操作");
        }
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
            std::cout<<'['<<'\n';
            for(int i=0;i<row;i++){
                const float *curr=p+i*col;
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
            std::cout<<'['<<'\n';
            for(int i=0;i<space;i++){
                std::cout<<'['<<'\n';
                for(int j=0;j<row;j++){
                    std::cout<<'['<<'\n';
                    const float *curr=p+i*row*col+j*col;
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

class MathOps{
public:
    static void add_tensor(std::shared_ptr<Tensor> input_a,std::shared_ptr<Tensor> input_b,std::shared_ptr<Tensor> output){
        size_t n=input_a->tensor_total_elements();
        float *a_ptr=static_cast<float *>(input_a->tensor_data_ptr());
        float *b_ptr=static_cast<float *>(input_b->tensor_data_ptr());
        float *out_ptr=static_cast<float *>(output->tensor_data_ptr());
        for(size_t i=0;i<n;i++){
            out_ptr[i]=a_ptr[i]+b_ptr[i];
        }
    }
    static void mul_tensor(std::shared_ptr<Tensor> input_a,std::shared_ptr<Tensor> input_b,std::shared_ptr<Tensor> output){
        size_t n=input_a->tensor_total_elements();
        float *a_ptr=static_cast<float *>(input_a->tensor_data_ptr());
        float *b_ptr=static_cast<float *>(input_b->tensor_data_ptr());
        float *out_ptr=static_cast<float *>(output->tensor_data_ptr());
        for(size_t i=0;i<n;i++){
            out_ptr[i]=a_ptr[i]*b_ptr[i];
        }
    }
    static void silu_tensor(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output){
        size_t n=input->tensor_total_elements();
        float *in_ptr=static_cast<float *>(input->tensor_data_ptr());
        float *out_ptr=static_cast<float *>(output->tensor_data_ptr());
        for(size_t i=0;i<n;i++){
            out_ptr[i]=in_ptr[i]/(1+std::exp(-in_ptr[i]));
        }
    }
    static void softmax_tensor(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output){
        std::vector<int> in_shapes=input->tensor_shapes();
        float *in_ptr=static_cast<float *>(input->tensor_data_ptr());
        float *out_ptr=static_cast<float *>(output->tensor_data_ptr());
        int n=in_shapes.size();
        int col=in_shapes[n-1];
        int row=input->tensor_total_elements()/col;
        for(int i=0;i<row;i++){
            float max_val=std::numeric_limits<float>::lowest();
            float sum=0;
            float *curr_in=in_ptr+i*col;
            float *curr_out=out_ptr+i*col;
            for(int j=0;j<col;j++){
                if(curr_in[j]>max_val){
                    max_val=curr_in[j];
                }
            }
            for(int j=0;j<col;j++){
                curr_out[j]=std::exp(curr_in[j]-max_val);
                sum=sum+curr_out[j];
            }
            float scale=1.0/sum;
            for(int j=0;j<col;j++){
                curr_out[j]=curr_out[j]*scale;
            }
        }
    }
    static void rope_tensor(std::shared_ptr<Tensor> q,std::shared_ptr<Tensor> k,ModelConfig& config){
        float *q_ptr=static_cast<float *>(q->tensor_data_ptr());
        float *k_ptr=static_cast<float *>(k->tensor_data_ptr());
        int heads=config.heads;
        int heads_dims= config.dims/heads;
        int curr_seq_len=q->tensor_shapes()[0];
        for(int i=0;i<curr_seq_len;i++){
            for(int j=0;j<heads;j++){
                for(int k=0;k<heads_dims;k=k+2){
                    float angle=i*1.0/std::pow(10000,(float )k/(float )heads_dims);
                    float angle_cos=std::cos(angle);
                    float angle_sin=std::sin(angle);
                    float *curr_q=q_ptr+i*heads*heads_dims+j*heads_dims+k;
                    float *curr_k=k_ptr+i*heads*heads_dims+j*heads_dims+k;
                    float q0=curr_q[0];
                    float q1=curr_q[1];
                    curr_q[0]=q0*angle_cos-q1*angle_sin;
                    curr_q[1]=q0*angle_sin+q1*angle_cos;
                    float k0=curr_k[0];
                    float k1=curr_k[1];
                    curr_k[0]=k0*angle_cos-k1*angle_sin;
                    curr_k[1]=k0*angle_sin+k1*angle_cos;
                }
            }
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

class RMSNormLayer:public Layer{
private:
    std::shared_ptr<Tensor> weights_;
    float eps_;
    ModelConfig config_;
public:
    RMSNormLayer(std::shared_ptr<Tensor> weights,ModelConfig& config,float eps=1e-5):weights_(weights),
                                eps_(eps), config_(config),Layer("RMSNorm"){}
    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override{
        std::vector<int> in_shapes=input->tensor_shapes();
        int curr_seq_len=in_shapes[0];
        int dims=config_.dims;
        float *in_ptr=static_cast<float *>(input->tensor_data_ptr());
        float *out_ptr=static_cast<float *>(output->tensor_data_ptr());
        float *w_ptr=static_cast<float *>(weights_->tensor_data_ptr());
        for(int i=0;i<curr_seq_len;i++){
            float sum=0;
            float *curr_in=in_ptr+i*dims;
            float *curr_out=out_ptr+i*dims;
            for(int j=0;j<dims;j++){
                sum=sum+curr_in[j]*curr_in[j];
            }
            float scale=1.0/std::sqrt(sum/dims+eps_);
            for(int j=0;j<dims;j++){
                curr_out[j]=curr_in[j]*scale*w_ptr[j];
            }
        }
    }
};

class LinearLayer:public Layer{
private:
    std::shared_ptr<Tensor> weights_;
public:
    LinearLayer(std::shared_ptr<Tensor> weights):weights_(weights), Layer("Linear"){}
    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override{
        float *in_ptr=static_cast<float *>(input->tensor_data_ptr());
        float *out_ptr=static_cast<float *>(output->tensor_data_ptr());
        float *w_ptr=static_cast<float *>(weights_->tensor_data_ptr());
        std::vector<int> in_shapes=input->tensor_shapes();
        std::vector<int> w_shapes=weights_->tensor_shapes();
        int K=in_shapes[in_shapes.size()-1];
        int M=input->tensor_total_elements()/K;
        int N=weights_->tensor_total_elements()/K;
        for(int i=0;i<M;i++){
            float *curr_in=in_ptr+i*K;
            float *curr_out=out_ptr+i*N;
            for(int j=0;j<N;j++){
                float sum=0;
                float *curr_w=w_ptr+j*K;
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
    std::shared_ptr<LinearLayer> gate_proj_;
    std::shared_ptr<LinearLayer> up_proj_;
    std::shared_ptr<LinearLayer> down_proj_;

    std::shared_ptr<Tensor> after_gate_;
    std::shared_ptr<Tensor> after_up_;

    ModelConfig config_;
public:
    SwiGLULayer(std::shared_ptr<LinearLayer> gate_proj,std::shared_ptr<LinearLayer> up_proj,
                std::shared_ptr<LinearLayer> down_proj,std::shared_ptr<Allocator> allocator,ModelConfig& config):
            Layer("SwiGLU"),gate_proj_(gate_proj),up_proj_(up_proj),down_proj_(down_proj), config_(config){
        int max_seq_len=config_.seq_len;
        int hidden_dims=config_.hidden_dims;
        after_gate_=std::make_shared<Tensor>(std::vector<int>{max_seq_len,hidden_dims},allocator);
        after_up_=std::make_shared<Tensor>(std::vector<int>{max_seq_len,hidden_dims},allocator);
    }
    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override{
        std::vector<int> in_shapes=input->tensor_shapes();
        int curr_seq_len=in_shapes[0];
        int hidden_dims=config_.hidden_dims;
        after_gate_->tensor_reshapes(std::vector<int>{curr_seq_len,hidden_dims});
        after_up_->tensor_reshapes(std::vector<int>{curr_seq_len,hidden_dims});
        gate_proj_->forward(input,after_gate_);
        MathOps::silu_tensor(after_gate_,after_gate_);
        up_proj_->forward(input,after_up_);
        MathOps::mul_tensor(after_gate_,after_up_,after_up_);
        down_proj_->forward(after_up_,output);
    }
};

class LlamaAttentionLayer:public Layer{
private:
    std::shared_ptr<LinearLayer> wq_;
    std::shared_ptr<LinearLayer> wk_;
    std::shared_ptr<LinearLayer> wv_;
    std::shared_ptr<LinearLayer> wo_;

    std::shared_ptr<Tensor> after_q_;
    std::shared_ptr<Tensor> after_k_;
    std::shared_ptr<Tensor> after_v_;
    std::shared_ptr<Tensor> after_qkt_;
    std::shared_ptr<Tensor> after_qktv_;

    ModelConfig config_;
public:
    LlamaAttentionLayer(std::shared_ptr<LinearLayer> wq,std::shared_ptr<LinearLayer> wk,
                        std::shared_ptr<LinearLayer> wv,std::shared_ptr<LinearLayer> wo,
                        ModelConfig& config,std::shared_ptr<Allocator> allocator):
            Layer("LlamaAttention"),wq_(wq),wk_(wk),wv_(wv),wo_(wo), config_(config){
        int max_seq_len=config_.seq_len;
        int dims=config.dims;
        int heads=config_.heads;
        after_q_=std::make_shared<Tensor>(std::vector<int>{max_seq_len,dims},allocator);
        after_k_=std::make_shared<Tensor>(std::vector<int>{max_seq_len,dims},allocator);
        after_v_=std::make_shared<Tensor>(std::vector<int>{max_seq_len,dims},allocator);
        after_qktv_=std::make_shared<Tensor>(std::vector<int>{max_seq_len,dims},allocator);
        after_qkt_=std::make_shared<Tensor>(std::vector<int>{heads,max_seq_len,max_seq_len},allocator);
    }
    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override{
        int dims=config_.dims;
        int heads=config_.heads;
        int heads_dims=dims/heads;
        std::vector<int> in_shapes=input->tensor_shapes();
        int curr_seq_len=in_shapes[0];
        std::vector<int> curr_shapes={curr_seq_len,dims};
        after_q_->tensor_reshapes(curr_shapes);
        after_k_->tensor_reshapes(curr_shapes);
        after_v_->tensor_reshapes(curr_shapes);
        after_qktv_->tensor_reshapes(curr_shapes);
        after_qkt_->tensor_reshapes(std::vector<int>{heads,curr_seq_len,curr_seq_len});
        wq_->forward(input,after_q_);
        wk_->forward(input,after_k_);
        MathOps::rope_tensor(after_q_,after_k_,config_);
        wv_->forward(input,after_v_);
        float *after_q_ptr=static_cast<float *>(after_q_->tensor_data_ptr());
        float *after_k_ptr=static_cast<float *>(after_k_->tensor_data_ptr());
        float *after_v_ptr=static_cast<float *>(after_v_->tensor_data_ptr());
        float *after_qkt_ptr=static_cast<float *>(after_qkt_->tensor_data_ptr());
        for(int i=0;i<heads;i++){
            for(int j=0;j<curr_seq_len;j++){
                float *curr_qkt=after_qkt_ptr+i*curr_seq_len*curr_seq_len+j*curr_seq_len;
                for(int k=0;k<curr_seq_len;k++){
                    if(k>j){
                        curr_qkt[k]=-1e9f;
                    }else{
                        float sum=0;
                        float *curr_q=after_q_ptr+i*heads_dims+j*dims;
                        float *curr_k=after_k_ptr+i*heads_dims+k*dims;
                        for(int sc=0;sc<heads_dims;sc++){
                            sum=sum+curr_q[sc]*curr_k[sc];
                        }
                        curr_qkt[k]=sum;
                    }
                }
            }
        }
        float scale=1.0/std::sqrt(heads_dims);
        size_t qkt_size=after_qkt_->tensor_total_elements();
        for(size_t i=0;i<qkt_size;i++){
            after_qkt_ptr[i]=after_qkt_ptr[i]*scale;
        }
        MathOps::softmax_tensor(after_qkt_,after_qkt_);
        float *after_qktv_ptr=static_cast<float *>(after_qktv_->tensor_data_ptr());
        for(int i=0;i<curr_seq_len;i++){
            for(int j=0;j<heads;j++){
                for(int k=0;k<heads_dims;k++){
                    float sum=0;
                    for(int sc=0;sc<curr_seq_len;sc++){
                        sum=sum+after_qkt_ptr[j*curr_seq_len*curr_seq_len+i*curr_seq_len+sc]*after_v_ptr[sc*dims+j*heads_dims+k];
                    }
                    after_qktv_ptr[i*dims+j*heads_dims+k]=sum;
                }
            }
        }
        wo_->forward(after_qktv_,output);
    }
};

class TransformerBlockLayer:public Layer{
private:
    std::shared_ptr<LlamaAttentionLayer> llamaAttentionLayer_;
    std::shared_ptr<SwiGLULayer> swiGluLayer_;
    std::shared_ptr<RMSNormLayer> rmsNormLayer_before_attention_;
    std::shared_ptr<RMSNormLayer> rmsNormLayer_before_swiglu_;

    std::shared_ptr<Tensor> after_attention_;
    std::shared_ptr<Tensor> after_swiglu_;
    std::shared_ptr<Tensor> after_rmsnorm_;

    ModelConfig config_;
public:
    TransformerBlockLayer(std::shared_ptr<LlamaAttentionLayer> llamaAttentionLayer,std::shared_ptr<SwiGLULayer> swiGluLayer,
                          std::shared_ptr<RMSNormLayer> rmsNormLayer_before_attention,std::shared_ptr<RMSNormLayer> rmsNormLayer_before_swiglu
                          ,ModelConfig& config,std::shared_ptr<Allocator> allocator):
                          Layer("TransformerBlock"),
                          llamaAttentionLayer_(llamaAttentionLayer),swiGluLayer_(swiGluLayer),rmsNormLayer_before_swiglu_(rmsNormLayer_before_swiglu),
                          rmsNormLayer_before_attention_(rmsNormLayer_before_attention), config_(config){
        int dims=config_.dims;
        int max_seq_len=config_.seq_len;
        after_attention_=std::make_shared<Tensor>(std::vector<int>{max_seq_len,dims},allocator);
        after_swiglu_=std::make_shared<Tensor>(std::vector<int>{max_seq_len,dims},allocator);
        after_rmsnorm_=std::make_shared<Tensor>(std::vector<int>{max_seq_len,dims},allocator);
    }
    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override{
        int curr_seq_len=input->tensor_shapes()[0];
        std::vector<int> curr_shapes={curr_seq_len,config_.dims};
        after_attention_->tensor_reshapes(curr_shapes);
        after_swiglu_->tensor_reshapes(curr_shapes);
        after_rmsnorm_->tensor_reshapes(curr_shapes);
        output->tensor_reshapes(curr_shapes);
        rmsNormLayer_before_attention_->forward(input,after_rmsnorm_);
        llamaAttentionLayer_->forward(after_rmsnorm_,after_attention_);
        MathOps::add_tensor(input,after_attention_,after_attention_);
        rmsNormLayer_before_swiglu_->forward(after_attention_,after_rmsnorm_);
        swiGluLayer_->forward(after_rmsnorm_,after_swiglu_);
        MathOps::add_tensor(after_attention_,after_swiglu_,output);
    }
};

class EmbeddingLayer:public Layer{
private:
    std::shared_ptr<Tensor> weights_;
    ModelConfig config_;
public:
    EmbeddingLayer(std::shared_ptr<Tensor> weights,ModelConfig& config): Layer("Embedding"),
                    weights_(weights), config_(config){}
    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output)override{}
    void forward_tokens(const std::vector<int>& tokens,std::shared_ptr<Tensor> output){
        int dims=config_.dims;
        float *out_ptr=static_cast<float *>(output->tensor_data_ptr());
        float *w_ptr=static_cast<float *>(weights_->tensor_data_ptr());
        int curr_seq_len=tokens.size();
        output->tensor_reshapes(std::vector<int>{curr_seq_len,dims});
        for(int i=0;i<curr_seq_len;i++){
            std::memcpy(out_ptr+i*dims,w_ptr+tokens[i]*dims,dims*sizeof (float ));
        }
    }

};

class LlamaModel:public Layer{
private:
    std::vector<std::shared_ptr<TransformerBlockLayer>> blocks_;
    std::shared_ptr<EmbeddingLayer> embeddingLayer_;
    std::shared_ptr<RMSNormLayer> rmsNormLayer_;
    std::shared_ptr<LinearLayer> lmhead_;

    std::shared_ptr<Tensor> hidden_state_1_;
    std::shared_ptr<Tensor> hidden_state_2_;

    ModelConfig config_;
public:
    LlamaModel(std::vector<std::shared_ptr<TransformerBlockLayer>> blocks,std::shared_ptr<EmbeddingLayer> embeddingLayer,
               std::shared_ptr<RMSNormLayer> rmsNormLayer,std::shared_ptr<LinearLayer> lmhead,ModelConfig& config,std::shared_ptr<Allocator> allocator):
            Layer("LlamaModel"),blocks_(blocks),embeddingLayer_(embeddingLayer),rmsNormLayer_(rmsNormLayer)
            ,lmhead_(lmhead), config_(config){
        int max_seq_len=config_.seq_len;
        int dims=config_.dims;
        hidden_state_1_=std::make_shared<Tensor>(std::vector<int>{max_seq_len,dims},allocator);
        hidden_state_2_=std::make_shared<Tensor>(std::vector<int>{max_seq_len,dims},allocator);
    }
    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override{}
    void forward_tokens(std::vector<int> tokens,std::shared_ptr<Tensor> output){
        int curr_seq_len=tokens.size();
        int dims=config_.dims;
        int vocab_sizes=config_.vocab_sizes;
        hidden_state_1_->tensor_reshapes(std::vector<int>{curr_seq_len,dims});
        hidden_state_2_->tensor_reshapes(std::vector<int>{curr_seq_len,dims});
        output->tensor_reshapes(std::vector<int>{curr_seq_len,vocab_sizes});
        embeddingLayer_->forward_tokens(tokens,hidden_state_1_);
        std::shared_ptr<Tensor> curr_in=hidden_state_1_;
        std::shared_ptr<Tensor> curr_out=hidden_state_2_;
        int layers=config_.layers;
        for(int i=0;i<layers;i++){
            blocks_[i]->forward(curr_in,curr_out);
            std::swap(curr_in,curr_out);
        }
        rmsNormLayer_->forward(curr_in,curr_out);
        lmhead_->forward(curr_out,output);
    }
};

class ModelLoader{

public:
    static std::shared_ptr<LlamaModel> load_model(const std::string& model_path,std::shared_ptr<Allocator> allocator){
        FILE* file=fopen(model_path.c_str(),"rb");
        if(!file){
            throw std::runtime_error("无法打开模型文件,请检查路径对不对!\n");
        }
        int header[7];
        fread(header,sizeof (int),7,file);
        ModelConfig config;
        config.dims=header[0];
        config.hidden_dims=header[1];
        config.layers=header[2];
        config.heads=header[3];
        config.kv_heads=header[4];
        config.vocab_sizes=header[5];
        config.seq_len=header[6];
        std::cout<<"✔成功读取ModelConfig配置\n";
        auto emb_w=std::make_shared<Tensor>(std::vector<int>{config.vocab_sizes,config.dims},allocator);
        std::vector<std::shared_ptr<Tensor>> attn_norm_ws,wq_ws,wk_ws,wv_ws,wo_ws;
        std::vector<std::shared_ptr<Tensor>> ffn_norm_ws,w1_ws,w2_ws,w3_ws;
        for(int i=0;i<config.layers;i++){
            attn_norm_ws.push_back(std::make_shared<Tensor>(std::vector<int>{config.dims},allocator));
            wq_ws.push_back(std::make_shared<Tensor>(std::vector<int>{config.dims,config.dims},allocator));
            wk_ws.push_back(std::make_shared<Tensor>(std::vector<int>{config.dims,config.dims},allocator));
            wv_ws.push_back(std::make_shared<Tensor>(std::vector<int>{config.dims,config.dims},allocator));
            wo_ws.push_back(std::make_shared<Tensor>(std::vector<int>{config.dims,config.dims},allocator));
            ffn_norm_ws.push_back(std::make_shared<Tensor>(std::vector<int>{config.dims},allocator));
            w1_ws.push_back(std::make_shared<Tensor>(std::vector<int>{config.hidden_dims,config.dims},allocator));
            w2_ws.push_back(std::make_shared<Tensor>(std::vector<int>{config.dims,config.hidden_dims},allocator));
            w3_ws.push_back(std::make_shared<Tensor>(std::vector<int>{config.hidden_dims,config.dims},allocator));
        }
        auto final_norm_w=std::make_shared<Tensor>(std::vector<int>{config.dims},allocator);
        auto lm_head_w=std::make_shared<Tensor>(std::vector<int>{config.vocab_sizes,config.dims},allocator);
        fread(emb_w->tensor_data_ptr(),sizeof (float ),emb_w->tensor_total_elements(),file);
        for(int i=0;i<config.layers;i++) fread(attn_norm_ws[i]->tensor_data_ptr(),sizeof (float ),config.dims,file);
        for(int i=0;i<config.layers;i++) fread(wq_ws[i]->tensor_data_ptr(),sizeof (float ),wq_ws[i]->tensor_total_elements(),file);
        for(int i=0;i<config.layers;i++) fread(wk_ws[i]->tensor_data_ptr(),sizeof (float ),wk_ws[i]->tensor_total_elements(),file);
        for(int i=0;i<config.layers;i++) fread(wv_ws[i]->tensor_data_ptr(),sizeof (float ),wv_ws[i]->tensor_total_elements(),file);
        for(int i=0;i<config.layers;i++) fread(wo_ws[i]->tensor_data_ptr(),sizeof (float ),wo_ws[i]->tensor_total_elements(),file);
        for(int i=0;i<config.layers;i++) fread(ffn_norm_ws[i]->tensor_data_ptr(),sizeof (float ),config.dims,file);
        for(int i=0;i<config.layers;i++) fread(w1_ws[i]->tensor_data_ptr(),sizeof (float ),w1_ws[i]->tensor_total_elements(),file);
        for(int i=0;i<config.layers;i++) fread(w2_ws[i]->tensor_data_ptr(),sizeof (float ),w2_ws[i]->tensor_total_elements(),file);
        for(int i=0;i<config.layers;i++) fread(w3_ws[i]->tensor_data_ptr(),sizeof (float ),w3_ws[i]->tensor_total_elements(),file);
        fread(final_norm_w->tensor_data_ptr(),sizeof (float ),final_norm_w->tensor_total_elements(),file);
        fread(lm_head_w->tensor_data_ptr(),sizeof (float ),lm_head_w->tensor_total_elements(),file);
        fclose(file);
        auto embedding=std::make_shared<EmbeddingLayer>(emb_w,config);
        std::vector<std::shared_ptr<TransformerBlockLayer>> blocks;
        for(int i=0;i<config.layers;i++){
            auto attn=std::make_shared<LlamaAttentionLayer>(
                    std::make_shared<LinearLayer>(wq_ws[i]),std::make_shared<LinearLayer>(wk_ws[i]),
                    std::make_shared<LinearLayer>(wv_ws[i]),std::make_shared<LinearLayer>(wo_ws[i]),
                            config,allocator);
            auto ffn=std::make_shared<SwiGLULayer>(
                    std::make_shared<LinearLayer>(w1_ws[i]),std::make_shared<LinearLayer>(w3_ws[i]),
                    std::make_shared<LinearLayer>(w2_ws[i]),allocator,config);
            auto norm1=std::make_shared<RMSNormLayer>(attn_norm_ws[i],config);
            auto norm2=std::make_shared<RMSNormLayer>(ffn_norm_ws[i],config);
            blocks.push_back(std::make_shared<TransformerBlockLayer>(attn,ffn,
                                                                     norm1,norm2,config,allocator));
        }
        auto final_norm=std::make_shared<RMSNormLayer>(final_norm_w,config);
        auto lm_head=std::make_shared<LinearLayer>(emb_w);
        return std::make_shared<LlamaModel>(blocks,embedding,final_norm,lm_head,config,allocator);
    }
};

class Tokenizer{
private:
    std::vector<std::string> vocab_;
    int vocab_size_;
public:
    Tokenizer(const std::string& tokenizer_path,int vocab_size):vocab_size_(vocab_size){
        FILE *file= fopen(tokenizer_path.c_str(),"rb");
        if(!file){
            throw std::runtime_error("无法打开Tokenizer文件!\n");
        }
        unsigned int max_token_length;
        if(fread(&max_token_length,sizeof (int),1,file)!=1){
            throw std::runtime_error("读取 max_token_length失败!\n");
        }
        for(int i=0;i<vocab_size_;i++){
            float score;
            fread(&score,sizeof (float ),1,file);
            int len;
            fread(&len,sizeof (int),1,file);
            std::vector<char> word_buf(len+1);
            fread(word_buf.data(),sizeof (char),len,file);
            word_buf[len]='\0';
            vocab_.push_back(std::string (word_buf.data()));
        }
        fclose(file);
        std::cout<<"✔tokenizer 成功载入\n";
    }
    std::string  decode(int token_id){
        if(token_id<0||token_id>=vocab_size_){
            return "";
        }
        return vocab_[token_id];
    }
};

int main() {
    std::cout << "🚀 正在启动吴航先的自研 LLaMA 推理引擎..." << std::endl;
    std::shared_ptr<Allocator> allocator = std::make_shared<CPUAllocator>();
    try{
        auto model=ModelLoader::load_model("models/stories15M.bin",allocator);
        Tokenizer tokenizer("models/tokenizer.bin",32000);
        std::vector<int> input_tokens={1};
        int max_generate_step=100;
        auto logits_output=std::make_shared<Tensor>(std::vector<int>{256,32000},allocator);
        std::cout<<"\n---------------- 故事开始 ----------------\n";
        auto start_time=std::chrono::high_resolution_clock ::now();
        for(int step=0;step<max_generate_step;step++){
            auto step_start=std::chrono::high_resolution_clock ::now();
            model->forward_tokens(input_tokens,logits_output);
            int curr_seq_len=input_tokens.size();
            float *logits_ptr=static_cast<float *>(logits_output->tensor_data_ptr());
            float *last_row_logits=logits_ptr+(curr_seq_len-1)*32000;
            int next_token_id=0;
            float max_val=last_row_logits[0];
            for(int i=1;i<32000;i++){
                if(last_row_logits[i]>max_val){
                    max_val=last_row_logits[i];
                    next_token_id=i;
                }
            }
            if(next_token_id==2) break;
            std::string word=tokenizer.decode(next_token_id);
            if(word=="<0x0A>"){word="\n";}
            auto step_end=std::chrono::high_resolution_clock ::now();
            auto step_ms=std::chrono::duration_cast<std::chrono::milliseconds>(step_end-step_start).count();
            if(word =="\n"){
                std::cout<<word<<std::flush;
            }else{
                std::cout<<word<<"\033[90m["<<1.0/step_ms*1000<<"tokens/"<<"s]\033[0m"<<'\n'<<std::flush;
            }
            input_tokens.push_back(next_token_id);
        }
        auto end_time=std::chrono::high_resolution_clock ::now();
        auto duration=std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time);
        std::cout<<"\n-----故事结束----\n";
        std::cout << "⏱️ 总耗时: " << duration.count()/1000.0 << " s\n";
        std::cout << "🚀 生成速度: " << (float)max_generate_step / (duration.count() / 1000.0f) << " tokens/秒\n";
    }catch(const std::exception&e){
        std::cerr << "\n❌ 糟糕，引擎启动失败: " << e.what() << '\n';
    }

    return 0;
}




