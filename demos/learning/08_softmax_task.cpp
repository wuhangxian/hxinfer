#include "cstring"
#include "cstdlib"
#include "memory"
#include "vector"
#include "numeric"
#include "iostream"
#include "algorithm"
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
    size_t buffer_byte_size() const{
        return byte_size_;
    }
    void* buffer_data_ptr(){
        return data_;
    }
    const void* buffer_data_ptr() const{
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
        total_elements_=std::accumulate(shapes_.begin(),shapes_.end(),1,std::multiplies<>());
        size_t byte_size=total_elements_*sizeof (float );
        buffer_=std::make_shared<Buffer>(byte_size,allocator);
    }
    const std::vector<int>& tensor_shape() const{
        return shapes_;
    }
    std::vector<int>& tensor_shape(){
        return shapes_;
    }
    const void* tensor_data_ptr() const{
        void* ptr=buffer_->buffer_data_ptr();
        return ptr;
    }
    size_t tensor_element_size(){
        return total_elements_;
    }
    void* tensor_data_ptr(){
        void* ptr=buffer_->buffer_data_ptr();
        return ptr;
    }
    void tensor_print_data(){
        int n=shapes_.size();
        float *p=(float *)tensor_data_ptr();
        if(n==1){
            std::cout<<"[";
            for(int i=0;i<shapes_[0];i++){
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
        if(n==3){
            int space=shapes_[0];
            int row=shapes_[1];
            int col=shapes_[2];
            std::cout<<"["<<std::endl;
            for(int i=0;i<space;i++){
                std::cout<<"[";
                for(int j=0;j<row;j++){
                    std::cout<<"[";
                    for(int k=0;k<col;k++){
                        std::cout<<p[i*row*col+j*col+k]<<" ";
                    }
                    std::cout<<"]";
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
    virtual void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output)=0;
    virtual ~Layer(){}
};

class softmaxLayer:public Layer{
public:
    softmaxLayer(): Layer("softmax"){}
    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override{
        std::vector<int> in_shapes=input->tensor_shape();
        int dims=in_shapes.back();
        int rows=(int)input->tensor_element_size()/dims;
        float *in_ptr=(float *) input->tensor_data_ptr();
        float *out_ptr=(float *)output->tensor_data_ptr();
        for(int i=0;i<rows;i++){
            float *current_in=in_ptr+i*dims;
            float *current_out=out_ptr+i*dims;
            float max_val=*std::max_element(current_in,current_in+dims);
            float sum=0.0f;
            for(int j=0;j<dims;j++){
                float exp_val=std::exp(current_in[j]-max_val);
                current_out[j]=exp_val;
                sum= sum+exp_val;
            }
            for(int j=0;j<dims;j++){
                current_out[j]=current_out[j]/sum;
            }
        }

    }
};

int main(){
    std::shared_ptr<Allocator> allocator=std::make_shared<CPUAllocator>();
    std::shared_ptr<Tensor> input=std::make_shared<Tensor>(std::vector<int>{2,2,3},allocator);
    std::shared_ptr<Tensor> output=std::make_shared<Tensor>(std::vector<int>{4,3},allocator);
    std::shared_ptr<softmaxLayer> softmax=std::make_shared<softmaxLayer>();
    float *in_ptr=(float *)input->tensor_data_ptr();
    in_ptr[0]=0;
    in_ptr[1]=1;
    in_ptr[2]=2;
    in_ptr[3]=3;
    in_ptr[4]=4;
    in_ptr[5]=5;
    in_ptr[6]=6;
    in_ptr[7]=7;
    in_ptr[8]=8;
    in_ptr[9]=9;
    in_ptr[10]=10;
    in_ptr[11]=11;
    std::cout<<"输入矩阵"<<std::endl;
    input->tensor_print_data();
    softmax->forward(input,output);
    std::cout<<"输出矩阵"<<std::endl;
    output->tensor_print_data();
}
