#include "cstring"
#include "cstdlib"
#include "memory"
#include "vector"
#include "numeric"
#include "iostream"
#include "cmath"
class Allocator{
public:
    virtual void* allocate(size_t byte_size)=0;
    virtual void release(void* ptr)=0;
    virtual ~Allocator(){}
};

class CPUAllocator:public Allocator{
public:
    void* allocate(size_t byte_size) override{
        void *ptr=malloc(byte_size);
        if(ptr== nullptr){
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
    void* data_;
public:
    Buffer(size_t byte_size,std::shared_ptr<Allocator> allocator):
            allocator_(allocator),byte_size_(byte_size){
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
    };
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
        total_elements_=std::accumulate(shapes_.begin(),shapes_.end(),
                                              1,std::multiplies<>());
        size_t byte_size=total_elements_*sizeof (float );
        buffer_=std::make_shared<Buffer>(byte_size,allocator);
    }
    const std::vector<int> & tensor_shapes() const{
        return shapes_;
    }
    size_t tensor_element_size(){
        return total_elements_;
    }
    void* tensor_data_ptr(){
        return buffer_->buffer_data_ptr();
    }
    const void* tensor_data_ptr() const{
        return buffer_->buffer_data_ptr();
    }
    void tensor_print_data() const{
        int n=shapes_.size();
        float *p=(float *)tensor_data_ptr();
        if(n==1){
            int len=shapes_[0];
            std::cout<<"[";
            for(int i=0;i<len;i++){
                std::cout<<p[i]<<" ";
            }
            std::cout<<"]"<<std::endl;
        }
        if(n==2){
            int row=shapes_[0];
            int col=shapes_[1];
            std::cout<<"["<<std::endl;
            for(int i=0;i<row;i++){
                std::cout<<"[";
                for(int j=0;j<col;j++){
                    std::cout<<p[i*col+j]<<" ";
                }
                std::cout<<"]"<<std::endl;
            }
            std::cout<<"]"<<std::endl;
        }
    }
};

class Layer{
public:
    std::string layer_name_;
    Layer(std::string layer_name):layer_name_(layer_name){}
    virtual void forward(std::shared_ptr<Tensor> input,
                         std::shared_ptr<Tensor> output)=0;
    virtual ~Layer(){}
};

class RMSNormLayer:public Layer{
private:
    std::shared_ptr<Tensor> weights_;
    int dim_;
    float eps_;
public:
    RMSNormLayer(int dim,std::shared_ptr<Allocator> allocator,float eps=1e-5)
                :Layer("RMSNorm"),dim_(dim),eps_(eps){
        weights_=std::make_shared<Tensor>(std::vector<int>{dim_},allocator);
        float *w_ptr=(float *)weights_->tensor_data_ptr();
        for(int i=0;i<dim_;i++){
            w_ptr[i]=1;
        }
    }
    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override{
        std::vector<int> in_shapes=input->tensor_shapes();
        int current_dim=in_shapes.back();
        if(current_dim!=dim_){
            std::cerr<<"RMSNorm层的可学习权重维度和输入维度不相符"<<std::endl;
            return;
        }
        int rows=input->tensor_element_size()/current_dim;
        float *in_ptr=(float *)input->tensor_data_ptr();
        float *out_ptr=(float *) output->tensor_data_ptr();
        float *w_ptr=(float *)weights_->tensor_data_ptr();
        for(int i=0;i<rows;i++){
            float *current_in=in_ptr+i*current_dim;
            float *current_out=out_ptr+i*current_dim;
            float sum_sq=0.0;
            for(int j=0;j<current_dim;j++){
                sum_sq=sum_sq+current_in[j]*current_in[j];
            }
            float mean_sum_sq=sum_sq/current_dim;
            float rms=std::sqrt(mean_sum_sq+eps_);
            for(int j=0;j<current_dim;j++){
                current_out[j]=current_in[j]/rms*w_ptr[j];
            }
        }
    }
};

int main(){
    std::shared_ptr<Allocator> allocator=std::make_shared<CPUAllocator>();
    std::shared_ptr<Tensor> input=std::make_shared<Tensor>(std::vector<int>{2,2},allocator);
    float *p=(float *)input->tensor_data_ptr();
    p[0]=3;
    p[1]=4;
    p[2]=1;
    p[3]=1;
    std::cout<<"---Input---"<<std::endl;
    input->tensor_print_data();
    std::shared_ptr<Tensor> output=std::make_shared<Tensor>(std::vector<int>{2,2},allocator);
    std::shared_ptr<RMSNormLayer> rmsnorm=std::make_shared<RMSNormLayer>(2,allocator);
    rmsnorm->forward(input,output);
    std::cout<<"---Output---"<<std::endl;
    output->tensor_print_data();
}