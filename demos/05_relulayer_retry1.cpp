#include "cstring"
#include "cstdlib"
#include "memory"
#include "vector"
#include "numeric"
#include "iostream"
class Allocator{
public:
    virtual void *allocate(size_t byte_size)=0;
    virtual void release(void *ptr)=0;
    virtual ~Allocator(){}
};

class CPUAllocator:public Allocator{
    void *allocate(size_t byte_size) override{
        void *ptr=malloc(byte_size);
        if(ptr== nullptr) return nullptr;
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
    Buffer(size_t type_size,std::shared_ptr<Allocator> allocator):
            byte_size_(type_size),allocator_(allocator){
        data_=allocator_->allocate(byte_size_);
    }
    ~Buffer(){
        if(data_!= nullptr){
            allocator_->release(data_);
            data_= nullptr;
        }
    }
    size_t buffer_size(){
        return byte_size_;
    }
    const void* buffer_ptr() const{
        return data_;
    }
};

class Tensor{
private:
    std::vector<int> shapes_;
    std::shared_ptr<Buffer> buffer_;
public:
    Tensor(std::vector<int> shapes,std::shared_ptr<Allocator> allocator):
                shapes_(shapes){
        int total_elements=std::accumulate(shapes_.begin(),shapes_.end(),1,std::multiplies<>());
        size_t byte_size=total_elements*sizeof (float );
        buffer_=std::make_shared<Buffer>(byte_size,allocator);
    }
    const std::vector<int>& tensor_shapes() const{
        return shapes_;
    }
    float * tensor_data_ptr() const{
        float *p=(float *)buffer_->buffer_ptr();
        if(p== nullptr) return nullptr;
        return p;
    }
    size_t tensor_element_count() const{
        if(buffer_!= nullptr) return buffer_->buffer_size()/sizeof (float );
        return 0;
    }
    float &tensor_at(int index){
        float *p=(float *)buffer_->buffer_ptr();
        return p[index];
    }

    void tensor_print(){
        if(buffer_== nullptr||shapes_.size()==0) return;
        float *p=tensor_data_ptr();
        if(shapes_.size()==1){
            int n=shapes_[0];
            std::cout<<"[";
            for(int i=0;i<n;i++){
                std::cout<<p[i]<<" ";
            }
            std::cout<<"]"<<std::endl;
        }
        if(shapes_.size()==2){
            int n=shapes_[0];
            int m=shapes_[1];
            std::cout<<"[";
            for(int i=0;i<n;i++){
                std::cout<<"[";
                for(int j=0;j<m;j++){
                    std::cout<<p[i*m+j]<<" ";
                }
                std::cout<<"]";
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

class ReluLayer:public Layer{
public:
    ReluLayer(): Layer("Relu"){}
    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output){
        if(input->tensor_element_count()!=output->tensor_element_count()){
            std::cerr<<"Size mismatch!"<<std::endl;
            return;
        }
        float *p_input=input->tensor_data_ptr();
        float *p_output=output->tensor_data_ptr();
        size_t count=input->tensor_element_count();

        for(int i=0;i<count;i++){
            float x=p_input[i];
            if(x>0) {
                p_output[i]=x;
            }else{
                p_output[i]=0;
            }
        }
    }
};

int main(){
    std::shared_ptr<Allocator> allocator=std::make_shared<CPUAllocator>();
    std::shared_ptr<Tensor> input=std::make_shared<Tensor>(std::vector<int>{2,3},allocator);
    std::shared_ptr<Tensor> output=std::make_shared<Tensor>(std::vector<int>{2,3},allocator);
    input->tensor_at(0)=1.0;
    input->tensor_at(1)=-1.0;
    input->tensor_at(2)=2.0;
    input->tensor_at(3)=-2.0;
    input->tensor_at(4)=3.0;
    input->tensor_at(5)=-3.0;
    std::shared_ptr<Layer> relulayer=std::make_shared<ReluLayer>();
    relulayer->forward(input,input);
    input->tensor_print();
}
