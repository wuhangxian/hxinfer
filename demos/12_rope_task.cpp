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
    virtual ~Allocator(){};
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
    Buffer operator= (const Buffer&)=delete;
    const size_t buffer_byte_size() const{
        return byte_size_;
    };
    const void* buffer_data_ptr() const{
        return data_;
    };
    void* buffer_data_ptr(){
        return  data_;
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
    virtual ~Tensor(){}
    const std::vector<int> &tensor_shape() const{
        return shapes_;
    }
    const size_t tensor_total_elements() const{
        return total_elements_;
    }
    const void* tensor_data_ptr() const{
        return buffer_->buffer_data_ptr();
    }
    void* tensor_data_ptr(){
        return buffer_->buffer_data_ptr();
    }
    void tensor_print_data(){
        int n=shapes_.size();
        float *p=(float *)tensor_data_ptr();
        if(n==1){
            int len=shapes_[0];
            std::cout<<"[";
            for(int i=0;i<len;i++){
                std::cout<<p[i]<<" ";
            }
            std::cout<<"]"<<'\n';
        }
        if(n==2){
            int row=shapes_[0];
            int col=shapes_[1];
            float *curr;
            std::cout<<"["<<'\n';
            for(int i=0;i<row;i++){
                curr=p+i*col;
                std::cout<<"[";
                for(int j=0;j<col;j++){
                    std::cout<<curr[j]<<" ";
                }
                std::cout<<"]"<<'\n';
            }
            std::cout<<"]"<<'\n';
        }
    }
};

class Layer{
public:
    std::string layer_name_;
    explicit Layer(std::string layer_name):layer_name_(layer_name){}
    virtual void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output)=0;
    virtual ~Layer(){}
};

class RoPELayer:public Layer{
private:
    int pos_;
public:
    RoPELayer(): Layer("RoPE"),pos_(0){}
    void set_pos(int pos){
        pos_=pos;
    }
    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override{
        size_t total_elements=input->tensor_total_elements();
        if(total_elements!=output->tensor_total_elements()){
            std::cerr<<"RoPE算子维度不匹配!"<<'\n';
            return;
        }
        float *in_ptr=(float *)input->tensor_data_ptr();
        float *out_ptr=(float *)output->tensor_data_ptr();
        int dim=total_elements;
        for(int i=0;i<dim;i=i+2){
            float x=in_ptr[i];
            float y=in_ptr[i+1];
            float freq=std::pow(10000,-((float )i/(float)dim));
            float theta=pos_*freq;
            float cos_theta=std::cos(theta);
            float sin_theta=std::sin(theta);
            out_ptr[i]=x*cos_theta-y*sin_theta;
            out_ptr[i+1]=x*sin_theta+y*cos_theta;
        }
    }
};

int main(){
    std::shared_ptr<Allocator> allocator=std::make_shared<CPUAllocator>();
    std::shared_ptr<Tensor> input=std::make_shared<Tensor>(std::vector<int>{4},allocator);
    std::shared_ptr<Tensor> output=std::make_shared<Tensor>(std::vector<int>{4},allocator);
    float *in_ptr=(float *)input->tensor_data_ptr();
    in_ptr[0]=1;in_ptr[1]=2;in_ptr[2]=3;in_ptr[3]=4;
    std::shared_ptr<RoPELayer> roPeLayer=std::make_shared<RoPELayer>();
    std::cout<<"-----input-----"<<'\n';
    input->tensor_print_data();
    roPeLayer->set_pos(0);
    roPeLayer->forward(input,output);
    std::cout << "\n--- pos = 0 旋转后的结果 (应该和输入一模一样，因为旋转0度) ---" << '\n';
    output->tensor_print_data();
    roPeLayer->set_pos(2);
    roPeLayer->forward(input,output);
    std::cout << "\n--- pos = 2 旋转后的结果 ---" << '\n';
    output->tensor_print_data();
    return 0;
}