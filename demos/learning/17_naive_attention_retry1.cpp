#include "cstring"
#include "cstdlib"
#include "iostream"
#include "memory"
#include "vector"
#include "numeric"
#include "cmath"
struct ModelConfig{
    int hidden_size;
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

class Layer{
public:
    std::string layer_name_;
    Layer(std::string layer_name):layer_name_(layer_name){}
    virtual void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output)=0;
    virtual ~Layer(){}
};

class LinearLayer:public Layer{
private:
    std::shared_ptr<Tensor> weights_;
    int in_features_;
    int out_features_;
public:
    LinearLayer(std::string name,int in_features,int out_features,std::shared_ptr<Allocator> allocator):
                    Layer(name),in_features_(in_features),out_features_(out_features){
        weights_=std::make_shared<Tensor>(std::vector<int>{out_features_,in_features_},allocator);
        weights_->tensor_fill_num(0.01);
    }
    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override{
        std::vector<int> in_size=input->tensor_shapes();
        std::vector<int> out_size=output->tensor_shapes();
        if(in_size[in_size.size()-1]!=in_features_){
            std::cerr<<"[Fatal]input输入的最后维度与权重矩阵的第一维度不相符\n";
            return;
        }
        if(out_size[out_size.size()-1]!=out_features_){
            std::cerr<<"[Fatal]权重矩阵的第二维度与output输出的最后维度不相符\n";
            return;
        }
        float *in_ptr=(float *)input->tensor_data_ptr();
        float *w_ptr=(float *)weights_->tensor_data_ptr();
        float *out_ptr=(float *)output->tensor_data_ptr();
        int K=in_features_;
        int M=input->tensor_total_elements()/K;
        int N=out_features_;
        for(int i=0;i<M;i++){
            for(int j=0;j<N;j++){
                float sum=0;
                for(int k=0;k<K;k++){
                    sum=sum+in_ptr[i*K+k]*w_ptr[j*K+k];
                }
                out_ptr[i*N+j]=sum;
            }
        }
    }
};

class AttentionBlock:public Layer{
private:
    std::shared_ptr<LinearLayer> q_proj_;
    std::shared_ptr<LinearLayer> k_proj_;
    std::shared_ptr<LinearLayer> v_proj_;
    std::shared_ptr<LinearLayer> o_proj_;
    int d_model_;
public:
    AttentionBlock(const ModelConfig& config,std::shared_ptr<Allocator> allocator):
                Layer("Attention"),d_model_(config.hidden_size){
        q_proj_=std::make_shared<LinearLayer>("q",d_model_,d_model_,allocator);
        k_proj_=std::make_shared<LinearLayer>("k",d_model_,d_model_,allocator);
        v_proj_=std::make_shared<LinearLayer>("v",d_model_,d_model_,allocator);
        o_proj_=std::make_shared<LinearLayer>("o",d_model_,d_model_,allocator);
    }
    void forward(std::shared_ptr<Tensor> input_X,std::shared_ptr<Tensor> output) override{
        int seq_len=input_X->tensor_shapes()[0];
        std::shared_ptr<Allocator> allocator=std::make_shared<CPUAllocator>();
        auto q_buffer=std::make_shared<Tensor>(std::vector<int>{seq_len,d_model_},allocator);
        auto k_buffer=std::make_shared<Tensor>(std::vector<int>{seq_len,d_model_},allocator);
        auto v_buffer=std::make_shared<Tensor>(std::vector<int>{seq_len,d_model_},allocator);
        auto ctx_buffer=std::make_shared<Tensor>(std::vector<int>{seq_len,d_model_},allocator);
        q_proj_->forward(input_X,q_buffer);
        k_proj_->forward(input_X,k_buffer);
        v_proj_->forward(input_X,v_buffer);
        float *q_ptr=(float *)q_buffer->tensor_data_ptr();
        float *k_ptr=(float *)k_buffer->tensor_data_ptr();
        float *v_ptr=(float *)v_buffer->tensor_data_ptr();
        float *ctx_ptr=(float *)ctx_buffer->tensor_data_ptr();
        float scale=1.0/std::sqrt(d_model_);
        for(int i=0;i<seq_len;i++){
            float max_s=-1e9f;
            std::vector<float> row_score(seq_len,0.0);
            for(int j=0;j<seq_len;j++){
                float sum=0;
                for(int k=0;k<d_model_;k++){
                    sum=sum+q_ptr[i*d_model_+k]*k_ptr[j*d_model_+k];
                }
                row_score[j]=sum*scale;
                if(row_score[j]>max_s){
                    max_s=row_score[j];
                }
            }
            float exp_sum=0;
            for(int j=0;j<seq_len;j++){
                row_score[j]=std::exp(row_score[j]-max_s);
                exp_sum=exp_sum+row_score[j];
            }
            for(int d=0;d<d_model_;d++){
                float sum=0;
                for(int j=0;j<seq_len;j++){
                    sum=sum+row_score[j]/exp_sum*v_ptr[j*d_model_+d];
                }
                ctx_ptr[i*d_model_+d]=sum;
            }

        }
        o_proj_->forward(ctx_buffer,output);
    }
};

int main(){
    std::cout<<"-----无KV Cache暴力推理-----\n";
    std::shared_ptr<Allocator>allocator=std::make_shared<CPUAllocator>();
    ModelConfig config;
    config.hidden_size=8;
    std::shared_ptr<AttentionBlock> attentionBlock=std::make_shared<AttentionBlock>(config,allocator);
    int prompt_size=3;
    for(int step=0;step<2;step++){
        int curr=prompt_size+step;
        std::cout << "[Step " << step << "] 输入给模型的句子长度: " << curr << '\n';
        auto input=std::make_shared<Tensor>(std::vector<int>{curr,config.hidden_size},allocator);
        auto output =std::make_shared<Tensor>(std::vector<int>{curr,config.hidden_size},allocator);
        input->tensor_fill_num(1);
        std::cout<<"---input---\n";
        input->tensor_print_data();
        attentionBlock->forward(input,output);
        std::cout<<"---output---\n";
        output->tensor_print_data();
    }
}


