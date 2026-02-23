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
    const std::vector<int>& tensor_shape(){
        return shapes_;
    }
    size_t tensor_total_elements(){
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
                std::cout<<'[';
                curr=p+i*col;
                for(int j=0;j<col;j++){
                    std::cout<<curr[j]<<' ';
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

class ArgmaxLayer:public Layer{
public:
    ArgmaxLayer(): Layer("Argmax"){}
    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override{
        size_t in_size=input->tensor_total_elements();
        size_t out_size=output->tensor_total_elements();
        if(out_size!=1){
            std::cout << "[Error] Argmax 输出 Tensor 必须大小为 1" << '\n';
            return;
        }
        float *in_ptr=(float *)input->tensor_data_ptr();
        float *out_ptr=(float *)output->tensor_data_ptr();
        float max_val=in_ptr[0];
        int max_index=0;
        for(int i=1;i<in_size;i++){
            if(in_ptr[i]>max_val){
                max_val=in_ptr[i];
                max_index=i;
            }
        }
        out_ptr[0]=(float )max_index;
    }
};
int main(){
    std::shared_ptr<Allocator> allocator=std::make_shared<CPUAllocator>();
    std::shared_ptr<Tensor> input=std::make_shared<Tensor>(std::vector<int> {6},allocator);
    std::shared_ptr<Tensor> output=std::make_shared<Tensor>(std::vector<int> {1},allocator);
    float *in_ptr=(float *)input->tensor_data_ptr();
    in_ptr[0]=0.1;in_ptr[1]=0.5;in_ptr[2]=0.9;
    in_ptr[3]=0.2;in_ptr[4]=0.2;in_ptr[5]=0.4;
    std::shared_ptr<ArgmaxLayer> argmaxLayer=std::make_shared<ArgmaxLayer>();
    argmaxLayer->forward(input,output);
    std::cout<<"-----input-----"<<'\n';
    input->tensor_print_data();
    std::cout<<"-----output-----"<<'\n';
    output->tensor_print_data();
}
