#include <cstring>
#include <cstdlib>
#include <memory>
#include <vector>
#include <numeric>
#include <iostream>
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
            std::cerr<<"未申请到地址"<<std::endl;
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
    Buffer(size_t byte_size,std::shared_ptr<Allocator> allocator):allocator_(allocator),byte_size_(byte_size){
        data_=allocator_->allocate(byte_size_);
    }

    ~Buffer(){
        if(data_!= nullptr){
            allocator_->release(data_);
            data_= nullptr;
        }
    }

    void* data(){
        return data_;
    }

    size_t byte_size() const{
        return byte_size_;
    }
};

class Tensor{
private:
    std::vector<int> shapes_;
    std::shared_ptr<Buffer> buffer_;

public:
    Tensor(std::vector<int> shapes,std::shared_ptr<Allocator> allocator):shapes_(shapes){
        int total_elements=std::accumulate(shapes_.begin(),shapes_.end(),1,std::multiplies<int>());
        size_t byte_size=total_elements*sizeof(float);
        buffer_=std::make_shared<Buffer>(byte_size,allocator);
    }
    //行为： 函数返回的是 ptr[index] 本身。没有发生复制，你拿到的就是内存里那个位置的“控制权”。
    //威力： 你既可以读取它，也可以直接修改它。
    //例子： at(0) = 10.0f; 是合法的。这行代码会直接把 buffer_ 对应位置的值改成 10.0。
    float& at(int index){
        float* ptr=(float*)buffer_->data();
        return ptr[index];
    }

    float* ptr(){
        if(buffer_){
            return (float *)buffer_->data();
        }
        return nullptr;
    }

    size_t size(){
        if(buffer_){
            return buffer_->byte_size()/sizeof(float );
        }
    }

    void print_data(){
        if(!buffer_) return;
        float* p=ptr();
        int count=size();
        std::cout << "[";
        for(int i=0;i<count;i++){
            std::cout<<p[i]<<" ";
        }
        std::cout<<"]"<<std::endl;
    }
};

class Layer{
public:
    std::string layer_name_;
    explicit Layer(const std::string& name):layer_name_(name){}

    virtual void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output)=0;

    virtual ~Layer(){}
};

class ReluLayer : public Layer{
public:
    ReluLayer(): Layer("Relu"){}

    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override{
        if(input->size()!=output->size()){
            std::cerr<<"报错,输入输出大小不一样,没法做Relu"<<std::endl;
            return ;
        }
        float *in_ptr=input->ptr();
        float *out_ptr=output->ptr();
        size_t total_count=input->size();

        std::cout<<"Layer["<<layer_name_<<"]正在计算..."<<std::endl;
        for(int i=0;i<total_count;i++){
            float x=in_ptr[i];
            if(x>0.0f){
                out_ptr[i]=x;
            }else{
                out_ptr[i]=0;
            }
        }
    }
};

int main(){
    std::shared_ptr<Allocator> allocator=std::make_shared<CPUAllocator>();
    std::shared_ptr<Tensor> tensor_in=std::make_shared<Tensor>(std::vector{2,3},allocator);
    tensor_in->at(0)=-2.0f;
    tensor_in->at(1)=2.0f;
    tensor_in->at(2)=-3.0f;
    tensor_in->at(3)=3.0f;
    tensor_in->at(4)=-4.0f;
    tensor_in->at(5)=4.0f;
    std::cout<<"--打印输入数据--"<<std::endl;
    tensor_in->print_data();
    std::shared_ptr<Tensor> tensor_out=std::make_shared<Tensor>(std::vector{2,3},allocator);
    std::shared_ptr<Layer> relu_layer=std::make_shared<ReluLayer>();
    relu_layer->forward(tensor_in,tensor_out);
    std::cout<<"--打印输出数据--"<<std::endl;
    tensor_out->print_data();
    return 0;
}




