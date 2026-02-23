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
    void *allocate(size_t byte_size) override{
        void *ptr=malloc(byte_size);
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
    Buffer(size_t byte_size,std::shared_ptr<Allocator> allocator):allocator_(allocator),
                byte_size_(byte_size){
        data_=allocator_->allocate(byte_size_);
    }
    ~Buffer(){
        if(data_!= nullptr){
            allocator_->release(data_);
            data_= nullptr;
        }
    }
    Buffer(const Buffer&)=delete;
    Buffer operator=(const Buffer&)=delete;
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
        total_elements_=std::accumulate(shapes_.begin(),shapes_.end(),
                                            1,std::multiplies<>());
        size_t byte_size=total_elements_*sizeof (float );
        buffer_=std::make_shared<Buffer>(byte_size,allocator);
    }
    size_t tensor_total_elements(){
        return total_elements_;
    }
    const std::vector<int> &tensor_shapes() const{
        return shapes_;
    };
    void* tenor_data_ptr(){
        return buffer_->buffer_data_ptr();
    }
    const void* tenor_data_ptr() const{
        return buffer_->buffer_data_ptr();
    }
    void tensor_print_data(){
        int n=shapes_.size();
        float *p=(float *)tenor_data_ptr();
        if(n==1){
            std::cout<<"[";
            int len=shapes_[0];
            for(int i=0;i<len;i++){
                std::cout<<p[i]<<" ";
            }
            std::cout<<"]"<<std::endl;
        }
        if(n==2){
            float *curr=p;
            std::cout<<"["<<std::endl;
            int row=shapes_[0];
            int col=shapes_[1];
            for(int i=0;i<row;i++){
                curr=p+i*col;
                std::cout<<"[";
                for(int j=0;j<col;j++){
                    std::cout<<curr[j]<<" ";
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
    explicit Layer(std::string layer_name):layer_name_(layer_name){}
    virtual void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output)=0;
    virtual ~Layer(){}
};

class SiLuLayer:public Layer{
public:
    SiLuLayer(): Layer("SiLu"){};
    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override{
        float *in_ptr=(float *)input->tenor_data_ptr();
        float *out_ptr=(float *)output->tenor_data_ptr();
        size_t in_size=input->tensor_total_elements();
        size_t out_size=output->tensor_total_elements();
        if(in_size!=out_size){
            std::cerr<<"输入与输出的维度不匹配"<<std::endl;
        }
        for(int i=0;i<in_size;i++){
            out_ptr[i]=in_ptr[i]/(1+exp(-1*in_ptr[i]));
        }
    }
};

int main(){
    std::shared_ptr<Allocator> allocator=std::make_shared<CPUAllocator>();
    std::shared_ptr<Tensor> input=std::make_shared<Tensor>(std::vector<int>{2,3},allocator);
    std::shared_ptr<SiLuLayer> siLuLayer=std::make_shared<SiLuLayer>();
    float *in_ptr=(float *)input->tenor_data_ptr();
    in_ptr[0]=-2.0;
    in_ptr[1]=-1.0;
    in_ptr[2]=0;
    in_ptr[3]=1.0;
    in_ptr[4]=2.0;
    in_ptr[5]=3.0;
    std::cout<<"输入的张量input"<<std::endl;
    input->tensor_print_data();
    siLuLayer->forward(input,input);
    std::cout<<"输出的张量output"<<std::endl;
    input->tensor_print_data();
}
