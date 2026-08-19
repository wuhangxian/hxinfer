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
            return nullptr;
        }
        memset(ptr,0,byte_size);
        return ptr;
    }
    void release(void* ptr) override{
            free(ptr);
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

    size_t byte_size(){
        return byte_size_;
    }

    void* data(){
        return data_;
    }
};

class Tensor{
private:
    std::vector<int> shapes_;
    std::shared_ptr<Buffer> buffer_;

public:
    Tensor(std::vector<int> shapes,std::shared_ptr<Allocator> allocator):shapes_(shapes){
        int total_elements=std::accumulate(shapes_.begin(),shapes_.end(),1,std::multiplies<int>());
        size_t byte_size=total_elements*sizeof (float );
        buffer_=std::make_shared<Buffer>(byte_size,allocator);
    }

    float &at(int index){
        float *ptr=(float *)buffer_->data();
        return ptr[index];
    }

    float *ptr(){
        if(buffer_!= nullptr){
            return (float *)buffer_->data();
        }
        return nullptr;
    }

    size_t size(){
        if(buffer_!= nullptr){
            return buffer_->byte_size()/sizeof(float );
        }
        return 0;
    }

    const std::vector<int>& get_shapes() const{
        return shapes_;
    }

    void print_data(){
        if(buffer_== nullptr) return;
        float *p=ptr();
        int count=size();
        std::cout<<"[";
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
    virtual ~Layer(){};
};

class LinearLayer:public Layer{
private:
    std::shared_ptr<Tensor> weights_;
    std::shared_ptr<Tensor> bias_;
    int in_features_;
    int out_features_;

public:
    LinearLayer(int in_features,int out_features,std::shared_ptr<Allocator> allocator):
            Layer("Linear"),in_features_(in_features),out_features_(out_features){
        weights_=std::make_shared<Tensor>(std::vector<int>{in_features_,out_features_},allocator);
        bias_=std::make_shared<Tensor>(std::vector<int>{out_features_},allocator);

    float *w_ptr=weights_->ptr();
    for(int i=0;i<weights_->size();i++){
        w_ptr[i]=1.0f;
    }
    weights_->print_data();
    float *b_ptr=bias_->ptr();
    for(int i=0;i<bias_->size();i++){
        b_ptr[i]=1.0f;
    }
    bias_->print_data();

    }

    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override{
        const std::vector<int>& in_shapes=input->get_shapes();
        int batch_size=in_shapes[0];
        int K=in_shapes[1];
        int N=out_features_;
        if(K!=in_features_){
            std::cerr<<"报错:输入数据的列数"<<K<<"与Linear层要求的"<<in_features_<<"不一致"<<std::endl;
            return;
        }
        std::cout << "Layer [" << layer_name_ << "] 开始计算 (矩阵乘法)..." << std::endl;
        float *in_ptr=input->ptr();
        float *out_ptr=output->ptr();
        float *w_ptr=weights_->ptr();
        float *b_ptr=bias_->ptr();
        for(int i=0;i<batch_size;i++){
            for(int j=0;j<N;j++){
                float sum=0;
                for(int k=0;k<K;k++){
                    sum=sum+in_ptr[i*K+k]*w_ptr[k*N+j];
                }
                sum=sum+b_ptr[j];
                out_ptr[i*N+j]=sum;
            }
        }

    }

};

int main(){
    std::shared_ptr<Allocator> allocator=std::make_shared<CPUAllocator>();
    std::shared_ptr<Tensor> tensor_in=std::make_shared<Tensor>(std::vector<int>{2,3},allocator);
    tensor_in->at(0)=1.0f; tensor_in->at(1)=1.0f; tensor_in->at(2)=1.0f;
    tensor_in->at(3)=2.0f; tensor_in->at(4)=2.0f; tensor_in->at(5)=2.0f;
    std::cout << "--- 输入数据 (2x3) ---" << std::endl;
    tensor_in->print_data();
    LinearLayer linear_layer(3, 2, allocator);
    std::shared_ptr<Tensor> tensor_out = std::make_shared<Tensor>(std::vector<int>{2, 2}, allocator);
    linear_layer.forward(tensor_in, tensor_out);
    std::cout << "--- 输出数据 (应该全是 4 和 7) ---" << std::endl;
    tensor_out->print_data();
    return 0;
}
