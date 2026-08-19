#include "cstring"
#include "cstdlib"
#include "iostream"
#include "memory"
#include "vector"
#include "numeric"
#include "cmath"
struct ModelConfig{
    int hidden_size=4;
    float eps=1e-5;
};
class Allocator{
public:
    virtual void* allocate(size_t byte_size)=0;
    virtual void release(void *ptr)=0;
    virtual ~Allocator(){}
};

class CPUAllocator:public Allocator{
public:
    void* allocate(size_t byte_size){
        void* ptr= nullptr;
        int ret= posix_memalign(&ptr,64,byte_size);
        if(ret!=0||ptr== nullptr){
            std::cerr<<"[Fatal]内存不够了,无法申请内存\n";
            return nullptr;
        }
        memset(ptr,0,byte_size);
        return ptr;
    }
    void release(void* ptr){
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
    Tensor(std::vector<int> shapes,std::shared_ptr<Allocator> allocator):shapes_(shapes){
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
        float *p=(float *)tensor_data_ptr();
        int n=shapes_.size();
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
            float *curr=p;
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
    virtual ~Layer(){}
};

class RMSNormLayer:public Layer{
private:
    std::shared_ptr<Tensor> weights_;
    int hidden_size_;
    float eps_;
public:
    RMSNormLayer(ModelConfig& config,std::shared_ptr<Allocator> allocator): Layer("RMSNorm"),
            hidden_size_(config.hidden_size),eps_(config.eps)
            {
        weights_=std::make_shared<Tensor>(std::vector<int>{hidden_size_},allocator);
        weights_->tensor_fill_num(0.1);
    }
    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override{
        int seq_len=input->tensor_total_elements()/hidden_size_;
        float *in_ptr=(float *)input->tensor_data_ptr();
        float *out_ptr=(float *)output->tensor_data_ptr();
        float *w_ptr=(float *)weights_->tensor_data_ptr();
        for(int i=0;i<seq_len;i++){
            float *curr_in=in_ptr+i*hidden_size_;
            float *curr_out=out_ptr+i*hidden_size_;
            float sum=0;
            for(int j=0;j<hidden_size_;j++){
                sum=sum+curr_in[j]*curr_in[j];
            }
            float RMS=1.0/std::sqrt(sum/hidden_size_+eps_);
            for(int j=0;j<hidden_size_;j++){
                curr_out[j]=curr_in[j]*RMS*w_ptr[j];
            }
        }
    }
};

class AddLayer:public Layer{
private:
    std::shared_ptr<Tensor> input_b_;
public:
    AddLayer(std::shared_ptr<Tensor>& input_b):input_b_(input_b),Layer("Add"){}
    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override{
        float *in_ptr=(float *)input->tensor_data_ptr();
        float *in_b_ptr=(float *)input_b_->tensor_data_ptr();
        float *out_ptr=(float *)output->tensor_data_ptr();
        for(size_t i=0;i<input->tensor_total_elements();i++){
            out_ptr[i]=in_ptr[i]+in_b_ptr[i];
        }
    }

};

int main(){
    std::shared_ptr<Allocator> allocator=std::make_shared<CPUAllocator>();
    std::shared_ptr<Tensor> input=std::make_shared<Tensor>(std::vector<int>{3,4},allocator);
    std::shared_ptr<Tensor> output=std::make_shared<Tensor>(std::vector<int>{3,4},allocator);
    ModelConfig config;
    float *in_ptr=(float *)input->tensor_data_ptr();
    in_ptr[0]=1;in_ptr[1]=2;in_ptr[2]=3;in_ptr[3]=4;
    in_ptr[4]=0;in_ptr[5]=0;in_ptr[6]=0;in_ptr[7]=0;
    in_ptr[8]=-1;in_ptr[9]=1;in_ptr[10]=-1;in_ptr[11]=1;
    std::shared_ptr<RMSNormLayer> rmsNormLayer=std::make_shared<RMSNormLayer>(config,allocator);
    std::cout<<"-----测试RMSNorm算子-----\n";
    std::cout<<"---input---\n";
    input->tensor_print_data();
    rmsNormLayer->forward(input,output);
    std::cout<<"---output---\n";
    output->tensor_print_data();
    std::cout<<"-----RMSNorm算子测试完毕-----\n";
    std::shared_ptr<Tensor> input_b=std::make_shared<Tensor>(std::vector<int>{3,4},allocator);
    float *in_b_ptr=(float *)input_b->tensor_data_ptr();
    in_b_ptr[0]=1;in_b_ptr[1]=2;in_b_ptr[2]=3;in_b_ptr[3]=4;
    in_b_ptr[4]=0;in_b_ptr[5]=0;in_b_ptr[6]=0;in_b_ptr[7]=0;
    in_b_ptr[8]=-1;in_b_ptr[9]=1;in_b_ptr[10]=-1;in_b_ptr[11]=1;
    std::cout<<"-----测试Add算子-----\n";
    std::cout<<"---input---\n";
    input->tensor_print_data();
    std::cout<<"---input_b---\n";
    input_b->tensor_print_data();
    std::shared_ptr<AddLayer> addLayer=std::make_shared<AddLayer>(input_b);
    addLayer->forward(input,output);
    std::cout<<"---output---\n";
    output->tensor_print_data();
    std::cout<<"-----Add算子测试完毕-----\n";

}

