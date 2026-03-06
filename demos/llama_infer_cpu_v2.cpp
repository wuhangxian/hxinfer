#include "cstring"
#include "cstdlib"
#include "iostream"
#include "memory"
#include "vector"
#include "numeric"
#include "cmath"
#include "limits"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/stat.h>


struct ModelConfig{
    int dim;
    int hidden_dim;
    int layer;
    int head;
    int kv_head;
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
        void *ptr= nullptr;
        int ret=posix_memalign(&ptr,64,byte_size);
        if(ret!=0||ptr== nullptr){
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
            byte_size_(byte_size),allocator_(allocator){
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
    size_t total_elements_;
    size_t byte_size_;
    std::shared_ptr<Buffer> buffer_;
public:
    Tensor(std::vector<int> shapes,std::shared_ptr<Allocator> allocator):
            shapes_(shapes){
        total_elements_=std::accumulate(shapes_.begin(),shapes_.end(),size_t{1},std::multiplies<>());
        byte_size_=sizeof (float )*total_elements_;
        buffer_=std::make_shared<Buffer>(byte_size_,allocator);
    }
    ~Tensor()=default;
    size_t tensor_total_elements() const{
        return total_elements_;
    }
    const std::vector<int>& tensor_shapes() const{
        return shapes_;
    }
    void tensor_reshape(std::vector<int> reshapes){
        size_t new_total_elements=std::accumulate(reshapes.begin(),reshapes.end(),
                                                  1,std::multiplies<>());
        size_t new_byte_size=sizeof (float )*new_total_elements;
        if(new_byte_size>byte_size_){
            throw std::runtime_error("内存大小不够!不可以进行reshape\n");
        }else{
            shapes_=reshapes;
            total_elements_=new_total_elements;
        }
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
                std::cout<<'[';
                const float* curr_p=p+i*col;
                for(int j=0;j<col;j++){
                    std::cout<<curr_p[j]<<' ';
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
                    const float* curr_p=p+i*row*col+j*col;
                    for(int k=0;k<col;k++){
                        std::cout<<curr_p[k]<<' ';
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
    static void add_tensor(std::shared_ptr<Tensor> input_a,std::shared_ptr<Tensor> input_b,std::shared_ptr<Tensor> output){
        float *in_a=static_cast<float *>(input_a->tensor_data_ptr());
        float *in_b=static_cast<float *>(input_b->tensor_data_ptr());
        float *out=static_cast<float *>(output->tensor_data_ptr());
        size_t total_elements_a=input_a->tensor_total_elements();
        size_t total_elements_b=input_b->tensor_total_elements();
        size_t total_elements_out=output->tensor_total_elements();
        if(total_elements_a!=total_elements_out||total_elements_b!=total_elements_out){
            throw std::runtime_error("Add算子的输入与输出维度不一致!无法操作\n");
        }else if(total_elements_a!=total_elements_b){
            throw std::runtime_error("Add算子的两个输入维度不一致!无法操作\n");
        }else{
            for(size_t i=0;i<total_elements_out;i++){
                out[i]=in_a[i]+in_b[i];
            }
        }
    }
    static void mul_tensor(std::shared_ptr<Tensor> input_a,std::shared_ptr<Tensor> input_b,std::shared_ptr<Tensor> output){
        float *in_a=static_cast<float *>(input_a->tensor_data_ptr());
        float *in_b=static_cast<float *>(input_b->tensor_data_ptr());
        float *out=static_cast<float *>(output->tensor_data_ptr());
        size_t total_elements_a=input_a->tensor_total_elements();
        size_t total_elements_b=input_b->tensor_total_elements();
        size_t total_elements_out=output->tensor_total_elements();
        if(total_elements_a!=total_elements_out||total_elements_b!=total_elements_out){
            throw std::runtime_error("Mul算子的输入与输出维度不一致!无法操作\n");
        }else if(total_elements_a!=total_elements_b){
            throw std::runtime_error("Mul算子的两个输入维度不一致!无法操作\n");
        }else{
            for(size_t i=0;i<total_elements_out;i++){
                out[i]=in_a[i]*in_b[i];
            }
        }
    }
    static void silu_tensor(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output){
        float *in=static_cast<float *>(input->tensor_data_ptr());
        float *out=static_cast<float *>(output->tensor_data_ptr());
        size_t total_elements_in=input->tensor_total_elements();
        size_t total_elements_out=output->tensor_total_elements();
        if(total_elements_in!=total_elements_out){
            throw std::runtime_error("Silu算子的输入与输出维度不匹配,无法操作\n");
        }else{
            for(size_t i=0;i<total_elements_out;i++){
                out[i]=in[i]/(1+std::exp(-in[i]));
            }
        }
    }
    static void softmax_tensor(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output){
        float *in=static_cast<float *>(input->tensor_data_ptr());
        float *out=static_cast<float *>(output->tensor_data_ptr());
        size_t total_elements_in=input->tensor_total_elements();
        size_t total_elements_out=output->tensor_total_elements();
        std::vector<int> in_shapes=input->tensor_shapes();
        int row=in_shapes[0];
        int col=in_shapes[1];
        if(total_elements_in!=total_elements_out){
            throw std::runtime_error("Softmax算子的输入与输出维度不匹配,无法操作\n");
        }else{
            for(int i=0;i<row;i++){
                float max_val=std::numeric_limits<float>::lowest();
                float *curr_in=in+i*col;
                float *curr_out=out+i*col;
                float sum=0;
                for(int j=0;j<col;j++){
                    if(curr_in[j]>max_val){
                        max_val=curr_in[j];
                    }
                }
                for(int j=0;j<col;j++){
                    curr_out[j]=std::exp(curr_in[j]-max_val);
                    sum=sum+curr_out[j];
                }
                for(int j=0;j<col;j++){
                    curr_out[j]=curr_out[j]/sum;
                }
            }
        }
    }
    static void argmax_tensor(std::shared_ptr<Tensor> input,int& output){
        output=0;
        size_t total_elements_input=input->tensor_total_elements();
        float *in=static_cast<float *>(input->tensor_data_ptr());
        float max_val=in[0];
        for(size_t i=1;i<total_elements_input;i++){
            if(in[i]>max_val){
                max_val=in[i];
                output=i;
            }
        }
    }
    static void scale_tensor(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output,ModelConfig& config){
        int head_dim=config.dim/config.head;
        float scale=1.0/std::sqrt(head_dim);
        float *in=static_cast<float *>(input->tensor_data_ptr());
        float *out=static_cast<float *>(output->tensor_data_ptr());
        size_t total_elements_input=input->tensor_total_elements();
        for(size_t i=0;i<total_elements_input;i++){
            out[i]=in[i]*scale;
        }
    }
    static void rmsnorm_tensor(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> weights,std::shared_ptr<Tensor> output,float eps=1e-5){
        float *in=static_cast<float *>(input->tensor_data_ptr());
        float *out=static_cast<float *>(output->tensor_data_ptr());
        float *w=static_cast<float *>(weights->tensor_data_ptr());
        std::vector<int> in_shapes=input->tensor_shapes();
        int row=in_shapes[0];
        int col=in_shapes[1];
        for(int i=0;i<row;i++){
            float *curr_in=in+i*col;
            float *curr_out=out+i*col;
            float sum=0;
            for(int j=0;j<col;j++){
                sum=sum+curr_in[j]*curr_in[j];
            }
            float scale=1.0f/(std::sqrt(sum/ col+eps));
            for(int j=0;j<col;j++){
                curr_out[j]=curr_in[j]*scale*w[j];
            }
        }
    }
    static void matmul_tensor(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> weights,std::shared_ptr<Tensor> output){
        float *in=static_cast<float *>(input->tensor_data_ptr());
        float *out=static_cast<float *>(output->tensor_data_ptr());
        float *w=static_cast<float *>(weights->tensor_data_ptr());
        std::vector<int> in_shapes=input->tensor_shapes();
        int K=in_shapes[in_shapes.size()-1];
        int M=input->tensor_total_elements()/K;
        int N=weights->tensor_total_elements()/K;
        for(int i=0;i<M;i++){
            float *curr_out=out+i*N;
            for(int j=0;j<N;j++){
                float sum=0;
                float *curr_in=in+i*K;
                float *curr_w=w+j*K;
                for(int k=0;k<K;k++){
                    sum=sum+curr_in[k]*curr_w[k];
                }
                curr_out[j]=sum;
            }
        }
    }
    static void rope_tensor(std::shared_ptr<Tensor> q,std::shared_ptr<Tensor> k,ModelConfig& config,int base=10000,int step=0){
        std::vector<int> q_shapes=q->tensor_shapes();
        int curr_seq_len=q_shapes[0];
        int head=config.head;
        int head_dim=config.dim/head;
        float *q_ptr=static_cast<float *>(q->tensor_data_ptr());
        float *k_ptr=static_cast<float *>(k->tensor_data_ptr());
        for(int i=0;i<curr_seq_len;i++){
            int pos=step+i;
            for(int j=0;j<head;j++){
                for(int k=0;k<head_dim;k=k+2){
                    float *curr_q=q_ptr+i*head*head_dim+j*head_dim+k;
                    float *curr_k=k_ptr+i*head*head_dim+j*head_dim+k;
                    float scale=1.0/std::pow(base,(float )k/(float )head_dim);
                    float angle=scale*pos;
                    float angle_cos=std::cos(angle);
                    float angle_sin=std::sin(angle);
                    float q0=curr_q[0]*angle_cos-curr_q[1]*angle_sin;
                    float q1=curr_q[0]*angle_sin+curr_q[1]*angle_cos;
                    curr_q[0]=q0;
                    curr_q[1]=q1;
                    float k0=curr_k[0]*angle_cos-curr_k[1]*angle_sin;
                    float k1=curr_k[0]*angle_sin+curr_k[1]*angle_cos;
                    curr_k[0]=k0;
                    curr_k[1]=k1;
                }
            }
        }
    }
    static void embedding_tensor(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> weights,std::shared_ptr<Tensor> output,ModelConfig& config){
        float *in=static_cast<float *>(input->tensor_data_ptr());
        float *w=static_cast<float *>(weights->tensor_data_ptr());
        float *out=static_cast<float *>(output->tensor_data_ptr());
        size_t curr_seq_len=input->tensor_total_elements();
        int dim=config.dim;
        for(int i=0;i<curr_seq_len;i++){
            int token_id=static_cast<int>(in[i]);
            float *curr_w=w+token_id*dim;
            float *curr_out=out+i*dim;
            std::memcpy(curr_out,curr_w,dim*sizeof (float ));
        }
    }
};

class RMSNormLayer:public Layer{
private:
    std::shared_ptr<Tensor> weight_;
    float eps_;
    ModelConfig config_;
public:
    RMSNormLayer(std::shared_ptr<Allocator> allocator,ModelConfig& config,float eps=1e-5):Layer("RMSNormLayer"),
            eps_(eps), config_(config){
        weight_=std::make_shared<Tensor>(std::vector<int>{config_.dim},allocator);
    }
    float *get_weight_ptr(){
        return static_cast<float *>(weight_->tensor_data_ptr());
    }
    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override{
        MathOps::rmsnorm_tensor(input,weight_,output,eps_);
    }
};

class LinearLayer:public Layer{
private:
    std::shared_ptr<Tensor> weights_;
    ModelConfig config_;
public:
    LinearLayer(std::shared_ptr<Allocator> allocator,int out_features,int in_features,ModelConfig& config):
            Layer("LineraLayer"), config_(config){
        weights_=std::make_shared<Tensor>(std::vector<int>{out_features,in_features},allocator);
    }
    LinearLayer(std::shared_ptr<Tensor> weights):Layer("LineraLayer"),weights_(weights){}
    float * get_weight_ptr(){
        return static_cast<float *>(weights_->tensor_data_ptr());
    }
    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override{
        MathOps::matmul_tensor(input,weights_,output);
    }
};

class EmbeddingLayer:public Layer{
private:
    std::shared_ptr<Tensor> weights_;
    ModelConfig config_;
public:
    EmbeddingLayer(std::shared_ptr<Allocator> allocator,ModelConfig& config):
            Layer("EmbeddingLayer"), config_(config){
        weights_=std::make_shared<Tensor>(std::vector<int>{config_.vocab_size,config_.dim},allocator);
    }
    float * get_weight_ptr(){
        return static_cast<float *>(weights_->tensor_data_ptr());
    }
    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override{
        MathOps::embedding_tensor(input,weights_,output,config_);
    }
};

class LlamaMLPlayer:public Layer{
private:
    std::shared_ptr<Tensor> gate_weight_;
    std::shared_ptr<Tensor> up_weight_;
    std::shared_ptr<Tensor> down_weight_;

    std::shared_ptr<Tensor> after_gate_;
    std::shared_ptr<Tensor> after_up_;

    ModelConfig config_;
    std::shared_ptr<Allocator> allocator_;
public:
    LlamaMLPlayer(std::shared_ptr<Allocator> allocator,ModelConfig& config):
                    Layer("LlamaMLPlayer"), config_(config),allocator_(allocator){
        gate_weight_=std::make_shared<Tensor>(std::vector<int>{config_.hidden_dim,config_.dim},allocator_);
        up_weight_=std::make_shared<Tensor>(std::vector<int>{config_.hidden_dim,config_.dim},allocator_);
        down_weight_=std::make_shared<Tensor>(std::vector<int>{config_.dim,config_.hidden_dim},allocator_);
        after_gate_=std::make_shared<Tensor>(std::vector<int>{config_.seq_len,config_.hidden_dim},allocator_);
        after_up_=std::make_shared<Tensor>(std::vector<int>{config_.seq_len,config_.hidden_dim},allocator_);
    }
    float *get_gate_weight_ptr(){
        return static_cast<float *>(gate_weight_->tensor_data_ptr());
    }
    float *get_up_weight_ptr(){
        return static_cast<float *>(up_weight_->tensor_data_ptr());
    }
    float *get_down_weight_ptr(){
        return static_cast<float *>(down_weight_->tensor_data_ptr());
    }
    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override{
        int curr_seq_len=input->tensor_total_elements()/config_.dim;
        int hidden_dim=config_.hidden_dim;
        std::vector<int> curr_shapes={curr_seq_len,hidden_dim};
        after_gate_->tensor_reshape(curr_shapes);
        after_up_->tensor_reshape(curr_shapes);
        std::shared_ptr<LinearLayer> gate_proj_=std::make_shared<LinearLayer>(gate_weight_);
        std::shared_ptr<LinearLayer> up_proj_=std::make_shared<LinearLayer>(up_weight_);
        std::shared_ptr<LinearLayer> down_proj_=std::make_shared<LinearLayer>(down_weight_);
        gate_proj_->forward(input,after_gate_);
        up_proj_->forward(input,after_up_);
        MathOps::silu_tensor(after_gate_,after_gate_);
        MathOps::mul_tensor(after_gate_,after_up_,after_up_);
        down_proj_->forward(after_up_,output);
    }
};

//不加KV Cache
class LlamaAttentionlayer:public Layer{
private:
    std::shared_ptr<Tensor> q_;
    std::shared_ptr<Tensor> k_;
    std::shared_ptr<Tensor> v_;
    std::shared_ptr<Tensor> o_;

    std::shared_ptr<Tensor> after_q_;
    std::shared_ptr<Tensor> after_k_;
    std::shared_ptr<Tensor> after_v_;
    std::shared_ptr<Tensor> after_qkt_;
    std::shared_ptr<Tensor> after_qktv_;

    ModelConfig config_;
public:
    LlamaAttentionlayer(std::shared_ptr<Allocator> allocator,ModelConfig config):
            Layer("LlamaAttentionlayer"), config_(config){
        std::vector<int> qkvo_shapes={config_.dim,config_.dim};
        std::vector<int> after_qkv_shapes={config_.seq_len,config_.dim};
        std::vector<int> after_qkt_shapes={config_.seq_len*config_.head,config_.seq_len};
        q_=std::make_shared<Tensor>(qkvo_shapes,allocator);
        k_=std::make_shared<Tensor>(qkvo_shapes,allocator);
        v_=std::make_shared<Tensor>(qkvo_shapes,allocator);
        o_=std::make_shared<Tensor>(qkvo_shapes,allocator);
        after_q_=std::make_shared<Tensor>(after_qkv_shapes,allocator);
        after_k_=std::make_shared<Tensor>(after_qkv_shapes,allocator);
        after_v_=std::make_shared<Tensor>(after_qkv_shapes,allocator);
        after_qktv_=std::make_shared<Tensor>(after_qkv_shapes,allocator);
        after_qkt_=std::make_shared<Tensor>(after_qkt_shapes,allocator);
    }
    float * get_q_ptr(){
        return static_cast<float *>(q_->tensor_data_ptr());
    }
    float * get_k_ptr(){
        return static_cast<float *>(k_->tensor_data_ptr());
    }
    float * get_v_ptr(){
        return static_cast<float *>(v_->tensor_data_ptr());
    }
    float * get_o_ptr(){
        return static_cast<float *>(o_->tensor_data_ptr());
    }
    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override{
        int curr_seq_len=input->tensor_total_elements()/config_.dim;
        int dim=config_.dim;
        int head=config_.head;
        int head_dim=dim/head;
        std::vector<int> curr_shapes={curr_seq_len,dim};
        std::shared_ptr<LinearLayer> q_proj_=std::make_shared<LinearLayer>(q_);
        std::shared_ptr<LinearLayer> k_proj_=std::make_shared<LinearLayer>(k_);
        std::shared_ptr<LinearLayer> v_proj_=std::make_shared<LinearLayer>(v_);
        std::shared_ptr<LinearLayer> o_proj_=std::make_shared<LinearLayer>(o_);
        after_q_->tensor_reshape(curr_shapes);
        after_k_->tensor_reshape(curr_shapes);
        after_v_->tensor_reshape(curr_shapes);
        q_proj_->forward(input,after_q_);
        k_proj_->forward(input,after_k_);
        v_proj_->forward(input,after_v_);
        //忘记加rope算子了
        MathOps::rope_tensor(after_q_,after_k_,config_);
        float *q_ptr=static_cast<float *>(after_q_->tensor_data_ptr());
        float *k_ptr=static_cast<float *>(after_k_->tensor_data_ptr());
        float *v_ptr=static_cast<float *>(after_v_->tensor_data_ptr());
        float *qkt_ptr=static_cast<float *>(after_qkt_->tensor_data_ptr());
        float *qktv_ptr=static_cast<float *>(after_qktv_->tensor_data_ptr());
        for(int i=0;i<head;i++){
            for(int j=0;j<curr_seq_len;j++){
                for(int k=0;k<curr_seq_len;k++){
                    float *curr_q=q_ptr+j*dim+i*head_dim;
                    float *curr_k=k_ptr+k*dim+i*head_dim;
                    float *curr_qkt=qkt_ptr+i*curr_seq_len*curr_seq_len+j*curr_seq_len+k;
                    if(j<k){
                        //出错
                        curr_qkt[0]=-1e9f;
                    }else{
                        float sum=0;
                        for(int sc=0;sc<head_dim;sc++){
                            sum=sum+curr_q[sc]*curr_k[sc];
                        }
                        curr_qkt[0]=sum;
                    }
                }
            }
        }
        MathOps::scale_tensor(after_qkt_,after_qkt_,config_);
        MathOps::softmax_tensor(after_qkt_,after_qkt_);
        for(int i=0;i<curr_seq_len;i++){
            for(int j=0;j<head;j++){
                for(int k=0;k<head_dim;k++){
                    float *curr_qkt=qkt_ptr+j*curr_seq_len*curr_seq_len+i*curr_seq_len;
                    float *curr_qktv=qktv_ptr+i*dim+j*head_dim+k;
                    //出错,粗心
                    float sum=0;
                    for(int sc=0;sc<curr_seq_len;sc++){
                        sum=sum+curr_qkt[sc]*v_ptr[sc*dim+j*head_dim+k];
                    }
                    curr_qktv[0]=sum;
                }

            }
        }
        o_proj_->forward(after_qktv_,output);
    }
};

//加KV Cache
class LlamaAttentionaddKVlayer:public Layer{
private:
    std::shared_ptr<Tensor> q_;
    std::shared_ptr<Tensor> k_;
    std::shared_ptr<Tensor> v_;
    std::shared_ptr<Tensor> o_;

    std::shared_ptr<Tensor> after_q_;
    std::shared_ptr<Tensor> after_k_;
    std::shared_ptr<Tensor> after_v_;
    std::shared_ptr<Tensor> after_qkt_;
    std::shared_ptr<Tensor> after_qktv_;

    ModelConfig config_;
    int step_=0;
    std::shared_ptr<Tensor> k_cache_;
    std::shared_ptr<Tensor> v_cache_;
public:
    LlamaAttentionaddKVlayer(std::shared_ptr<Allocator> allocator,ModelConfig& config):
            Layer("LlamaAttentionaddKVlayer"), config_(config){
        std::vector<int> qkvo_shapes={config_.dim,config_.dim};
        std::vector<int> after_qkv_shapes={1,config_.dim};
        std::vector<int> after_qkt_shapes={config_.head*config_.dim,config_.dim};
        std::vector<int> cache_shapes={config_.seq_len,config_.dim};

        q_=std::make_shared<Tensor>(qkvo_shapes,allocator);
        k_=std::make_shared<Tensor>(qkvo_shapes,allocator);
        v_=std::make_shared<Tensor>(qkvo_shapes,allocator);
        o_=std::make_shared<Tensor>(qkvo_shapes,allocator);
        after_q_=std::make_shared<Tensor>(after_qkv_shapes,allocator);
        after_k_=std::make_shared<Tensor>(after_qkv_shapes,allocator);
        after_v_=std::make_shared<Tensor>(after_qkv_shapes,allocator);
        after_qkt_=std::make_shared<Tensor>(after_qkt_shapes,allocator);
        after_qktv_=std::make_shared<Tensor>(after_qkv_shapes,allocator);
        k_cache_=std::make_shared<Tensor>(cache_shapes,allocator);
        v_cache_=std::make_shared<Tensor>(cache_shapes,allocator);
    }
    float *get_q_ptr(){
        return static_cast<float *>(q_->tensor_data_ptr());
    }
    float *get_k_ptr(){
        return static_cast<float *>(k_->tensor_data_ptr());
    }
    float *get_v_ptr(){
        return static_cast<float *>(v_->tensor_data_ptr());
    }
    float *get_o_ptr(){
        return static_cast<float *>(o_->tensor_data_ptr());
    }
    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override{
        std::shared_ptr<LinearLayer> q_proj_=std::make_shared<LinearLayer>(q_);
        std::shared_ptr<LinearLayer> k_proj_=std::make_shared<LinearLayer>(k_);
        std::shared_ptr<LinearLayer> v_proj_=std::make_shared<LinearLayer>(v_);
        std::shared_ptr<LinearLayer> o_proj_=std::make_shared<LinearLayer>(o_);
        q_proj_->forward(input,after_q_);
        k_proj_->forward(input,after_k_);
        v_proj_->forward(input,after_v_);
        MathOps::rope_tensor(after_q_,after_k_,config_);
        float *k_ptr=static_cast<float *>(after_k_->tensor_data_ptr());
        float *v_ptr=static_cast<float *>(after_v_->tensor_data_ptr());
        float *q_ptr=static_cast<float *>(after_q_->tensor_data_ptr());
        float *k_cache_ptr=static_cast<float *>(k_cache_->tensor_data_ptr());
        float *v_cache_ptr=static_cast<float *>(v_cache_->tensor_data_ptr());
        int dim=config_.dim;
        int head=config_.head;
        int head_dim=dim/head;
        if(step_>config_.seq_len){
            throw std::runtime_error("OOM!\n");
        }
        std::memcpy(k_cache_ptr+step_*dim,k_ptr,dim*sizeof (float ));
        std::memcpy(v_cache_ptr+step_*dim,v_ptr,dim*sizeof (float ));
        int total_history=step_+1;
        std::vector<float> attention_scores(total_history,0);
        std::vector<float> current_out(dim,0);
        float scale=1.0/std::sqrt((float )head_dim);
        for(int i=0;i<head;i++){
            float *curr_q=q_ptr+i*head_dim;
            for(int j=0;j<total_history;j++){
                float *curr_k=k_cache_ptr+j*dim+i*head_dim;
                float sum=0;
                for(int k=0;k<head_dim;k++){
                    sum=sum+curr_q[k]*curr_k[k];
                }
                attention_scores[j]=sum*scale;
            }
            float max_val=std::numeric_limits<float>::lowest();
            for(int j=0;j<total_history;j++){
                if(attention_scores[j]>max_val){
                    max_val=attention_scores[j];
                }
            }
            float sum_exp=0;
            for(int j=0;j<total_history;j++){
                attention_scores[j]=std::exp(attention_scores[j]-max_val);
                sum_exp=sum_exp+attention_scores[j];
            }
            for(int j=0;j<total_history;j++){
                attention_scores[j]=attention_scores[j]/sum_exp;
            }
            for(int j=0;j<head_dim;j++){
                float sum=0;
                for(int k=0;k<total_history;k++){
                    sum=sum+attention_scores[k]*v_cache_ptr[k*dim+i*head_dim+j];
                }
                current_out[i*head_dim+j]=sum;
            }
        }
        float *out_ptr=static_cast<float *>(after_qktv_->tensor_data_ptr());
        std::memcpy(out_ptr,current_out.data(),dim*sizeof (float ));
        o_proj_->forward(after_qktv_,output);
        step_++;
    }
};


// ========================================================================
// 积木 1：LlamaBlock (把 Attention 和 MLP 用残差连接拼起来！)
// ========================================================================
class LlamaBlock : public Layer {
private:
    std::shared_ptr<RMSNormLayer> attention_norm_;
    std::shared_ptr<LlamaAttentionaddKVlayer> attention_;
    std::shared_ptr<RMSNormLayer> ffn_norm_;
    std::shared_ptr<LlamaMLPlayer> mlp_;
    std::shared_ptr<Tensor> hidden_; // 用于存放中间结果

    ModelConfig config_;
    std::shared_ptr<Allocator> allocator_;

public:
    LlamaBlock(std::shared_ptr<Allocator> allocator, ModelConfig& config)
            : Layer("LlamaBlock"), config_(config), allocator_(allocator) {
        attention_norm_ = std::make_shared<RMSNormLayer>(allocator, config);
        attention_ = std::make_shared<LlamaAttentionaddKVlayer>(allocator, config);
        ffn_norm_ = std::make_shared<RMSNormLayer>(allocator, config);
        mlp_ = std::make_shared<LlamaMLPlayer>(allocator, config);

        // 临时变量，每次最多处理 seq_len 个词
        hidden_ = std::make_shared<Tensor>(std::vector<int>{config.seq_len, config.dim}, allocator);
    }

    // 暴露出权重指针，方便 Loader 灌入数据
    float* get_attn_norm_ptr() { return attention_norm_->get_weight_ptr(); }
    float* get_wq_ptr() { return attention_->get_q_ptr(); }
    float* get_wk_ptr() { return attention_->get_k_ptr(); }
    float* get_wv_ptr() { return attention_->get_v_ptr(); }
    float* get_wo_ptr() { return attention_->get_o_ptr(); }

    float* get_ffn_norm_ptr() { return ffn_norm_->get_weight_ptr(); }
    float* get_w1_ptr() { return mlp_->get_gate_weight_ptr(); }
    float* get_w2_ptr() { return mlp_->get_down_weight_ptr(); }
    float* get_w3_ptr() { return mlp_->get_up_weight_ptr(); }

    void forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output) override {
        int curr_seq_len = input->tensor_total_elements() / config_.dim;
        hidden_->tensor_reshape({curr_seq_len, config_.dim});

        // 1. Attention 分支：hidden = Attention(RMSNorm(input))
        attention_norm_->forward(input, hidden_);
        attention_->forward(hidden_, hidden_);

        // 残差连接 1：input = input + hidden
        MathOps::add_tensor(input, hidden_, input);

        // 2. MLP 分支：hidden = MLP(RMSNorm(input))
        ffn_norm_->forward(input, hidden_);
        mlp_->forward(hidden_, hidden_);

        // 残差连接 2：output = input + hidden
        MathOps::add_tensor(input, hidden_, output);
    }
};

// ========================================================================
// 积木 2：完整的 LLaMA 模型 (组装流水线)
// ========================================================================
class LlamaModel : public Layer {
public:
    std::shared_ptr<EmbeddingLayer> embedding_;
    std::vector<std::shared_ptr<LlamaBlock>> blocks_;
    std::shared_ptr<RMSNormLayer> final_norm_;
    std::shared_ptr<LinearLayer> lm_head_;

    ModelConfig config_;
    std::shared_ptr<Allocator> allocator_;

    LlamaModel(std::shared_ptr<Allocator> allocator, ModelConfig& config)
            : Layer("LlamaModel"), config_(config), allocator_(allocator) {

        embedding_ = std::make_shared<EmbeddingLayer>(allocator, config);
        for (int i = 0; i < config.layer; i++) {
            blocks_.push_back(std::make_shared<LlamaBlock>(allocator, config));
        }
        final_norm_ = std::make_shared<RMSNormLayer>(allocator, config);
        lm_head_ = std::make_shared<LinearLayer>(allocator, config.vocab_size, config.dim, config);
    }

    void forward(std::shared_ptr<Tensor> input_tokens, std::shared_ptr<Tensor> output_logits) override {
        int curr_seq_len = input_tokens->tensor_total_elements();
        auto hidden = std::make_shared<Tensor>(std::vector<int>{curr_seq_len, config_.dim}, allocator_);

        // 1. 查字典，变成词向量
        embedding_->forward(input_tokens, hidden);

        // 2. 依次穿过所有 Transformer Block
        for (int i = 0; i < config_.layer; i++) {
            blocks_[i]->forward(hidden, hidden);
        }

        // 3. 最后的归一化
        final_norm_->forward(hidden, hidden);

        // 4. 预测下一个词的概率分布 (lm_head)
        lm_head_->forward(hidden, output_logits);
    }
};

// ========================================================================
// 积木 3：二进制文件 Loader (使用工业级 mmap 内存映射)
// ========================================================================
class ModelLoader {
public:
    static void load_weights(std::string path, std::shared_ptr<LlamaModel> model) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("找不到模型文件: " + path);
        struct stat sb; fstat(fd, &sb);
        float* data = static_cast<float*>(mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
        if (data == MAP_FAILED) throw std::runtime_error("mmap 失败");

        ModelConfig* file_cfg = reinterpret_cast<ModelConfig*>(data);
        std::cout << "--- 从二进制文件解析出的真实模型配置 ---" << std::endl;
        std::cout << "Dim: " << file_cfg->dim << ", Hidden_dim: " << file_cfg->hidden_dim << std::endl;
        std::cout << "Layers: " << file_cfg->layer << ", Heads: " << file_cfg->head << std::endl;
        std::cout << "Vocab: " << file_cfg->vocab_size << std::endl;

        // 强转安全读取，严格按物理排序载入
        float* w = data + (sizeof(ModelConfig) / sizeof(float));
        ModelConfig& cfg = model->config_;
        int head_dim = cfg.dim / cfg.head;

        std::memcpy(model->embedding_->get_weight_ptr(), w, cfg.vocab_size * cfg.dim * sizeof(float)); w += cfg.vocab_size * cfg.dim;
        for (int i = 0; i < cfg.layer; i++) { std::memcpy(model->blocks_[i]->get_attn_norm_ptr(), w, cfg.dim * sizeof(float)); w += cfg.dim; }
        for (int i = 0; i < cfg.layer; i++) { std::memcpy(model->blocks_[i]->get_wq_ptr(), w, cfg.dim * cfg.dim * sizeof(float)); w += cfg.dim * cfg.dim; }
        for (int i = 0; i < cfg.layer; i++) { int kv_dim = cfg.kv_head * head_dim; std::memcpy(model->blocks_[i]->get_wk_ptr(), w, kv_dim * cfg.dim * sizeof(float)); w += kv_dim * cfg.dim; }
        for (int i = 0; i < cfg.layer; i++) { int kv_dim = cfg.kv_head * head_dim; std::memcpy(model->blocks_[i]->get_wv_ptr(), w, kv_dim * cfg.dim * sizeof(float)); w += kv_dim * cfg.dim; }
        for (int i = 0; i < cfg.layer; i++) { std::memcpy(model->blocks_[i]->get_wo_ptr(), w, cfg.dim * cfg.dim * sizeof(float)); w += cfg.dim * cfg.dim; }
        for (int i = 0; i < cfg.layer; i++) { std::memcpy(model->blocks_[i]->get_ffn_norm_ptr(), w, cfg.dim * sizeof(float)); w += cfg.dim; }
        for (int i = 0; i < cfg.layer; i++) { std::memcpy(model->blocks_[i]->get_w1_ptr(), w, cfg.hidden_dim * cfg.dim * sizeof(float)); w += cfg.hidden_dim * cfg.dim; }
        for (int i = 0; i < cfg.layer; i++) { std::memcpy(model->blocks_[i]->get_w2_ptr(), w, cfg.dim * cfg.hidden_dim * sizeof(float)); w += cfg.dim * cfg.hidden_dim; }
        for (int i = 0; i < cfg.layer; i++) { std::memcpy(model->blocks_[i]->get_w3_ptr(), w, cfg.hidden_dim * cfg.dim * sizeof(float)); w += cfg.hidden_dim * cfg.dim; }

// 3. Final Norm
        std::memcpy(model->final_norm_->get_weight_ptr(), w, cfg.dim * sizeof(float));
        w += cfg.dim;

        // ==========================================================
        // 🚀 抢救 1：修正 RoPE 的指针跳过量 (实部+虚部刚好等于 1 倍)
        // ==========================================================
        w += cfg.seq_len * head_dim;

        // ==========================================================
        // 🚀 抢救 2：拦截 EOF 越界，实现真正的“权重共享”！
        // ==========================================================
        if (file_cfg->vocab_size > 0) {
            std::cout << "[Info] 命中权重共享机制！直接复用 Embedding 内存！" << std::endl;
            // 绝不往下读文件了！直接把开头读到的 Embedding 拷贝给 LM_Head！
            std::memcpy(model->lm_head_->get_weight_ptr(),
                        model->embedding_->get_weight_ptr(),
                        cfg.vocab_size * cfg.dim * sizeof(float));
        } else {
            std::cout << "[Info] 读取独立的 LM Head 权重..." << std::endl;
            std::memcpy(model->lm_head_->get_weight_ptr(), w, cfg.vocab_size * cfg.dim * sizeof(float));
        }

        munmap(data, sb.st_size);
        close(fd);
        std::cout << "✅ stories15M.bin 权重完美加载完毕！引擎就绪！" << std::endl;
    } // load_weights 函数结束
};

// ========================================================================
// 积木 4：解码器 Tokenizer (把冷冰冰的数字变成人话)
// ========================================================================
class Tokenizer {
private:
    std::vector<std::string> vocab_;
public:
    Tokenizer(std::string path, int vocab_size) {
        FILE *file = fopen(path.c_str(), "rb");
        if (!file) throw std::runtime_error("找不到 tokenizer.bin 文件！");

        int max_token_length;
        if (fread(&max_token_length, sizeof(int), 1, file) != 1) throw std::runtime_error("读取tokenizer失败");

        for (int i = 0; i < vocab_size; i++) {
            float score;
            if (fread(&score, sizeof(float), 1, file) != 1) break;
            int len;
            if (fread(&len, sizeof(int), 1, file) != 1) break;

            std::string word(len, ' '); // 预分配字符串空间
            if (fread(&word[0], 1, len, file) != len) break;

            vocab_.push_back(word);
        }
        fclose(file);
        std::cout << "✅ tokenizer.bin 字典加载成功！准备说人话！" << std::endl;
    }

    std::string decode(int token_id) {
        if (token_id >= 0 && token_id < vocab_.size()) {
            std::string text = vocab_[token_id];
            // LLaMA 的特殊字符处理：把代表空格的 ' ' 替换回真正的空格
            if (text.length() >= 3 && text.substr(0, 3) == " ") { // 这里的 " " 是 LLaMA 里的特殊下划线 _ (U+2581)
                text = " " + text.substr(3);
            }
            return text;
        }
        return "";
    }
};

// ========================================================================
// 积木 4：点火主程序！(先读 Config，再建引擎)
// ========================================================================
#include <chrono> // 必须加上这个头文件！

// ========================================================================
// 积木 4：点火主程序！(带硬核测速仪)
// ========================================================================
int main() {
    std::cout << "====== 🚀 正在启动吴航先的自研 LLaMA 推理引擎 ======" << std::endl;
    try {
        std::string model_path = "models/stories15M.bin";
        std::string tokenizer_path = "models/tokenizer.bin";

        // 1. 提取 Config
        int fd = open(model_path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("找不到模型文件!");
        ModelConfig config;
        read(fd, &config, sizeof(ModelConfig));
        close(fd);

        // 2. 实例化引擎
        std::shared_ptr<Allocator> allocator = std::make_shared<CPUAllocator>();
        auto model = std::make_shared<LlamaModel>(allocator, config);

        // 3. 灌入权重
        ModelLoader::load_weights(model_path, model);

        // 4. 加载解码器
        Tokenizer tokenizer(tokenizer_path, config.vocab_size);

        std::cout << "\n---------------- 故事开始 ----------------\n";

        int current_token_id = 1; // 1 通常是 <s> BOS
        int max_generate_step = 100;

        // 记录全局开始时间
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int step = 0; step < max_generate_step; step++) {
            // 记录单步开始时间
            auto step_start = std::chrono::high_resolution_clock::now();

            auto input_token = std::make_shared<Tensor>(std::vector<int>{1}, allocator);
            static_cast<float*>(input_token->tensor_data_ptr())[0] = static_cast<float>(current_token_id);
            auto logits = std::make_shared<Tensor>(std::vector<int>{1, config.vocab_size}, allocator);

            // ⚡ 核心前向传播！
            model->forward(input_token, logits);

            int next_token_id = 0;
            MathOps::argmax_tensor(logits, next_token_id);

            // 解码 Token
            std::string word = tokenizer.decode(next_token_id);
            if (word.find("<0x0A>") != std::string::npos) {
                word = "\n";
            }

            // 记录单步结束时间（使用微秒避免除以0）
            auto step_end = std::chrono::high_resolution_clock::now();
            auto step_us = std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start).count();
            float tokens_per_sec = 1000000.0f / (step_us == 0 ? 1 : step_us);

            // 极客风终端输出：带灰色高亮帧率
            if (word == "\n") {
                std::cout << word << std::flush;
            } else {
                // \033[90m 是终端打印灰色字体，\033[0m 是恢复默认颜色
                std::cout << word << "\033[90m[" << tokens_per_sec << " tk/s]\033[0m" <<'\n'<< std::flush;
            }

            current_token_id = next_token_id;
        }

        // 计算全局时间
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        std::cout << "\n\n---------------- 故事结束 ----------------\n";
        std::cout << "⏱️  总耗时: " << duration_ms / 1000.0f << " 秒\n";
        std::cout << "🚀 平均生成速度: " << (float)max_generate_step / (duration_ms / 1000.0f) << " tokens/秒\n";

    } catch (const std::exception& e) {
        std::cerr << "❌ 引擎崩溃: " << e.what() << std::endl;
    }
    return 0;
}

