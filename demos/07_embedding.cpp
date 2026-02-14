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
    size_t buffer_byte_size(){
        return byte_size_;
    }
    void* buffer_ptr(){
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
        int total_elements=std::accumulate(shapes_.begin(),shapes_.end(),
                                           1,std::multiplies<int>());
        size_t byte_size=total_elements*sizeof (float );
        buffer_=std::make_shared<Buffer>(byte_size,allocator);
    }
    float &tensor_at(int index){
        float *ptr=(float *)buffer_->buffer_ptr();
        return ptr[index];
    }
    float *tensor_ptr() const{
        return (float *)buffer_->buffer_ptr();
    }
    size_t tensor_element_size() const{
        return buffer_->buffer_byte_size()/sizeof (float );
    }
    const std::vector<int>& tensor_shapes(){
        return shapes_;
    }
    void tensor_print_matrix(){
        if(buffer_== nullptr||shapes_.empty()) return;
        float *p=tensor_ptr();
        if(shapes_.size()==1){
            std::cout<<"[";
            for(int i=0;i<shapes_[0];i++){
                std::cout<<p[i]<<" ";
            }
            std::cout<<"]"<<std::endl;
        }
        if(shapes_.size()==2){
            std::cout<<"["<<std::endl;
            for(int i=0;i<shapes_[0];i++){
                std::cout<<"[";
                for(int j=0;j<shapes_[1];j++){
                    std::cout<<p[i*shapes_[1]+j]<<" ";
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
    explicit Layer(const std::string& layer_name):layer_name_(layer_name){}
    virtual void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output)=0;
    virtual ~Layer(){}
};

class EmbeddingLayer:public Layer{
private:
    std::shared_ptr<Tensor> weights_;
    int vocab_size_;
    int embedding_dim_;
public:
    EmbeddingLayer(int vocab_size,int embedding_dim,std::shared_ptr<Allocator> allocator)
                : Layer("Embedding"),vocab_size_(vocab_size),embedding_dim_(embedding_dim){
        weights_=std::make_shared<Tensor>(std::vector<int>{vocab_size_,embedding_dim_},
                                          allocator);
        float *w_ptr=weights_->tensor_ptr();
        for(int i=0;i<vocab_size_;i++){
            for(int j=0;j<embedding_dim_;j++) {
                w_ptr[i*embedding_dim_+j]=(float )i+0.1*j;
            }
        }

    }
    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override {
        const std::vector<int> &in_shapes = input->tensor_shapes();
        int seq_len = in_shapes[0];
        float *in_ptr = input->tensor_ptr();
        float *out_ptr = output->tensor_ptr();
        float *w_ptr = weights_->tensor_ptr();
        std::cout << "Layer[" << layer_name_ << "]正在疯狂查字典..." << std::endl;
        for (int i = 0; i < seq_len; i++) {
            int token_id = (int) in_ptr[i];
            if (token_id < 0 || token_id >= vocab_size_) {
                std::cerr << "致命错误:输入的单词ID=" << token_id <<
                          "超出了字典总数" << vocab_size_ << "!" << std::endl;
                continue;
            }
            for (int j = 0; j < embedding_dim_; j++) {
                out_ptr[i * embedding_dim_ + j] = w_ptr[token_id * embedding_dim_ + j];
            }
        }
    }
};

int main(){
    std::shared_ptr<Allocator> allocator=std::make_shared<CPUAllocator>();
    EmbeddingLayer embeddingLayer(10,4,allocator);
    std::shared_ptr<Tensor> tensor_input=std::make_shared<Tensor>
            (std::vector<int>{3},allocator);
    tensor_input->tensor_at(0)=5;
    tensor_input->tensor_at(1)=0;
    tensor_input->tensor_at(2)=2;
    std::cout<<"-----用户的输入(Token IDs)-----"<<std::endl;
    tensor_input->tensor_print_matrix();
    std::shared_ptr<Tensor> tensor_output=std::make_shared<Tensor>
            (std::vector<int>{3,4},allocator);
    embeddingLayer.forward(tensor_input,tensor_output);
    std::cout<<"-----进行转换后的词嵌入矩阵-----"<<std::endl;
    tensor_output->tensor_print_matrix();
    return 0;
}


