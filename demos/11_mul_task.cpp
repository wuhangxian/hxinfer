#include "cstring"
#include "cstdlib"
#include "memory"
#include "vector"
#include "numeric"
#include "iostream"
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

class MulLayer:public Layer{
private:
    std::shared_ptr<Tensor> input_b_;
public:
    MulLayer(std::shared_ptr<Tensor> input_b):Layer("Mul"),input_b_(input_b){}
    void forward(std::shared_ptr<Tensor> input_a,std::shared_ptr<Tensor> output) override{
        size_t in_a_ele=input_a->tensor_total_elements();
        size_t in_b_ele=input_b_->tensor_total_elements();
        size_t out_ele=output->tensor_total_elements();
        if(in_a_ele!=in_b_ele||in_a_ele!=out_ele||in_b_ele!=out_ele){
            std::cout<<"输入与输出的Tensor维度不匹配"<<'\n';
            return;
        }
        float *in_a_ptr=(float *)input_a->tensor_data_ptr();
        float *in_b_ptr=(float *)input_b_->tensor_data_ptr();
        float *out_ptr=(float *)output->tensor_data_ptr();
        for(int i=0;i<out_ele;i++){
            out_ptr[i]=in_a_ptr[i]*in_b_ptr[i];
        }
    }
};

int main(){
    std::shared_ptr<Allocator> allocator=std::make_shared<CPUAllocator>();
    std::shared_ptr<Tensor> input_a=std::make_shared<Tensor>(std::vector{2,3},allocator);
    std::shared_ptr<Tensor> input_b=std::make_shared<Tensor>(std::vector{2,3},allocator);
    std::shared_ptr<Tensor> output=std::make_shared<Tensor>(std::vector{2,3},allocator);
    float *in_a=(float *)input_a->tensor_data_ptr();
    float *in_b=(float *)input_b->tensor_data_ptr();
    in_a[0]=1;in_a[1]=2;in_a[2]=3;
    in_a[3]=4;in_a[4]=5;in_a[5]=6;
    in_b[0]=0.5;in_b[1]=0.5;in_b[2]=0.5;
    in_b[3]=2;in_b[4]=2;in_b[5]=2;
    std::shared_ptr<MulLayer> mulLayer=std::make_shared<MulLayer>(input_b);
    std::cout<<"-----input_a-----"<<'\n';
    input_a->tensor_print_data();
    std::cout<<"-----input_b-----"<<'\n';
    input_b->tensor_print_data();
    mulLayer->forward(input_a,output);
    std::cout<<"-----output-----"<<'\n';
    output->tensor_print_data();
}