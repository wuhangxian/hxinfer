#include "cstring"
#include "cstdlib"
#include "memory"
#include "vector"
#include "numeric"
#include "iostream"
#include "cmath"
struct ModelConfig{
    int hidden_size;
    int intermediate_size;
    int num_hidden_layers;
    int vocab_size;
};

class Allocator{
public:
    virtual void* allocate(size_t byte_size)=0;
    virtual void release(void* ptr)=0;
    virtual ~Allocator(){}
};

class CPUAllocator:public Allocator{
public:
    void* allocate(size_t byte_size) override{
        void* ptr=malloc(byte_size);
        if(ptr== nullptr){
            return nullptr;
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
    void *data_;
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
    size_t buffer_byte_size(){
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
        total_elements_=std::accumulate(shapes_.begin(),shapes_.end()
                                        ,1,std::multiplies<>());
        size_t byte_size=total_elements_*sizeof (float );
        buffer_=std::make_shared<Buffer>(byte_size,allocator);
    }
    ~Tensor(){}
    size_t tensor_total_elements(){
        return total_elements_;
    }
    const std::vector<int>& tensor_shape(){
        return shapes_;
    }
    const void *tensor_data_ptr() const{
        return buffer_->buffer_data_ptr();
    }
    void *tensor_data_ptr(){
        return buffer_->buffer_data_ptr();
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
            int row=shapes_[0];
            int col=shapes_[1];
            std::cout<<'['<<'\n';
            float *curr=p;
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
    void tensor_fill_data(float val){
        float *p=(float *)tensor_data_ptr();
        for(size_t i=0;i<total_elements_;i++){
            p[i]=val;
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
    std::shared_ptr<Tensor> weight_;
    int in_features_;
    int out_features_;
public:
    LinearLayer(std::string name,int in_features,int out_features,
                std::shared_ptr<Allocator> allocator):
            Layer(name),in_features_(in_features),out_features_(out_features){
        weight_=std::make_shared<Tensor>(std::vector<int>{out_features_,in_features_},allocator);
        weight_->tensor_fill_data(0.01);
    }

    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override{
        float *in_ptr=(float *)input->tensor_data_ptr();
        float *w_ptr=(float *)weight_->tensor_data_ptr();
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

class SiLuLayer:public Layer{
public:
    SiLuLayer(): Layer("SiLu"){}
    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override{
        size_t in_size=input->tensor_total_elements();
        size_t out_size=output->tensor_total_elements();
        float *in_ptr=(float *)input->tensor_data_ptr();
        float *out_ptr=(float *)output->tensor_data_ptr();
        if(in_size!=out_size){
            std::cerr<<"SiLu算子输入与输出的张量维度不匹配";
            return;
        }
        for(int i=0;i<in_size;i++){
            out_ptr[i]=in_ptr[i]/(1+std::exp(-in_ptr[i]));
        }
    }
};

class MulLayer:public Layer{
private:
    std::shared_ptr<Tensor> input_b_;
public:
    MulLayer(std::shared_ptr<Tensor> input_b): Layer("Mul"),input_b_(input_b){}
    void forward(std::shared_ptr<Tensor> input_a,std::shared_ptr<Tensor> output) override{
        size_t input_size=input_a->tensor_total_elements();
        float *b_ptr=(float *)input_b_->tensor_data_ptr();
        float *a_ptr=(float *)input_a->tensor_data_ptr();
        float *out_ptr=(float *)output->tensor_data_ptr();
        for(size_t i=0;i<input_size;i++){
            out_ptr[i]=a_ptr[i]*b_ptr[i];
        }
    }
};

class SwiGLUBlock{
private:
    std::shared_ptr<LinearLayer> gate_proj_;
    std::shared_ptr<LinearLayer> up_proj_;
    std::shared_ptr<LinearLayer> down_proj_;
    std::shared_ptr<SiLuLayer> silu_;
    std::shared_ptr<MulLayer> mul_;

    std::shared_ptr<Tensor> gate_out_buf_;
    std::shared_ptr<Tensor> up_out_buf_;
    std::shared_ptr<Tensor> hidden_out_buf_;

public:
    SwiGLUBlock(const ModelConfig&config,std::shared_ptr<Allocator> allocator){
        int d_model=config.hidden_size;
        int inter_dim=config.intermediate_size;

        std::vector<int> inter_shape={1,inter_dim};
        gate_out_buf_=std::make_shared<Tensor>(inter_shape,allocator);
        up_out_buf_=std::make_shared<Tensor>(inter_shape,allocator);
        hidden_out_buf_=std::make_shared<Tensor>(inter_shape,allocator);

        gate_proj_=std::make_shared<LinearLayer>("Gate",d_model,inter_dim,allocator);
        up_proj_=std::make_shared<LinearLayer>("Up",d_model,inter_dim,allocator);
        down_proj_=std::make_shared<LinearLayer>("down",inter_dim,d_model,allocator);

        silu_=std::make_shared<SiLuLayer>();
        mul_=std::make_shared<MulLayer>(up_out_buf_);

    }
    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output){
        gate_proj_->forward(input,gate_out_buf_);
        up_proj_->forward(input,up_out_buf_);
        silu_->forward(gate_out_buf_,gate_out_buf_);
        mul_->forward(gate_out_buf_,hidden_out_buf_);
        down_proj_->forward(hidden_out_buf_,output);
    }
};

int main() {
    std::shared_ptr<Allocator> alloc = std::make_shared<CPUAllocator>();

    // 1. 模拟解析 config.json (这里我们依然用小数字测试，但架构已经是真实的了)
    ModelConfig config;
    config.hidden_size = 4;
    config.intermediate_size = 8;
    config.num_hidden_layers = 1;
    config.vocab_size = 100;

    // 2. 准备输入输出
    std::shared_ptr<Tensor> input = std::make_shared<Tensor>(std::vector<int>{1, config.hidden_size}, alloc);
    std::shared_ptr<Tensor> output = std::make_shared<Tensor>(std::vector<int>{1, config.hidden_size}, alloc);
    input->tensor_fill_data(1.0f); // 塞满 1.0 测试

    // 3. 实例化整个 SwiGLU，把 config 传进去
    std::shared_ptr<SwiGLUBlock> swiglu = std::make_shared<SwiGLUBlock>(config, alloc);

    std::cout << "--- 真实维度动态初始化的 SwiGLU 启动 ---" << '\n';
    swiglu->forward(input, output);

    std::cout << "\n--- 输出结果 ---" << '\n';
    output->tensor_print_data();

    return 0;
}

