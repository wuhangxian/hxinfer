#include "cstring"
#include "cstdlib"
#include "memory"
#include "vector"
#include "numeric"
#include "iostream"
class Allocator{
public:
    virtual void *allocate(size_t byte_size)=0;
    virtual void release(void* ptr)=0;
    virtual ~Allocator(){}
};

class CPUAllocator:public Allocator{
public:
    void* allocate(size_t byte_size) override{
        void *ptr=malloc(byte_size);
        if(ptr== nullptr){
            return ptr;
        }
        memset(ptr,0,byte_size);
        return ptr;
    }
    void release(void* ptr){
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
        data_= allocator_->allocate(byte_size_);
    }
    ~Buffer(){
        if(data_!= nullptr){
            allocator_->release(data_);
            data_= nullptr;
        }
    }
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
    size_t buffer_byte_size ()const{
        return byte_size_;
    }
    const void *buffer_data_ptr() const{
        return data_;
    }
    void *buffer_data_ptr(){
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
        size_t total_elements=std::accumulate(shapes_.begin(),shapes_.end(),
                                           1,std::multiplies<>());
        size_t byte_size=total_elements*sizeof(float );
        buffer_=std::make_shared<Buffer>(byte_size,allocator);
    }

    const std::vector<int>& tensor_shapes(){
        return shapes_;
    };
    const void* tensor_data_ptr() const{
        return buffer_->buffer_data_ptr();
    }
    void* tensor_data_ptr(){
        return buffer_->buffer_data_ptr();
    }
    size_t tensor_element_size(){
        return std::accumulate(shapes_.begin(),shapes_.end(),
                               1,std::multiplies<>());
    }
    void tensor_print_data(){
        int n=shapes_.size();
        float *tensor_p=(float *)tensor_data_ptr();
        if(n==1){
            std::cout<<"[";
            for(int i=0;i<shapes_[0];i++){
                std::cout<<tensor_p[i]<<" ";
            }
            std::cout<<"]"<<std::endl;
        }
        if(n==2){
            int row=shapes_[0];
            int col=shapes_[1];
            std::cout<<"[";
            for(int i=0;i<row;i++){
                std::cout<<"[";
                for(int j=0;j<col;j++){
                    std::cout<<tensor_p[i*col+j]<<" ";
                }
                std::cout<<"]"<<std::endl;
            }
            std::cout<<"]"<<std::endl;
        }
        if(n==3){
            int space=shapes_[0];
            int row=shapes_[1];
            int col=shapes_[2];
            std::cout<<"[";
            for(int i=0;i<space;i++){
                std::cout<<"[";
                for(int j=0;j<row;j++){
                    std::cout<<"[";
                    for(int k=0;k<col;k++){
                        std::cout<<tensor_p[i*row*col+j*col+k]<<" ";
                    }
                    std::cout<<"]"<<std::endl;
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
    explicit Layer(std::string name):layer_name_(name){}
    virtual void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output)=0;
    virtual ~Layer(){}
};

class EmbeddingLayer:public Layer{
private:
    std::shared_ptr<Tensor> vocab_;
    int vocab_size_;
    int embed_dims_;
public:
    std::shared_ptr<Tensor>& embedding_vocab(){
        return vocab_;
    }
    const std::shared_ptr<Tensor>& embedding_vocab() const{
        return vocab_;
    }
    EmbeddingLayer(int vocab_size,int embed_dims,
                   std::shared_ptr<Allocator> allocator):Layer("embedding")
                   ,vocab_size_(vocab_size),embed_dims_(embed_dims){
        vocab_=std::make_shared<Tensor>(std::vector<int>{vocab_size_,embed_dims_},allocator);
        float * vocab_p=(float *)vocab_->tensor_data_ptr();
        int row=vocab_size_;
        int col=embed_dims_;
        for(int i=0;i<row;i++){
            for(int j=0;j<col;j++){
                vocab_p[i*col+j]=i+0.1*j;
            }
        }
    }
    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output){
        const std::vector<int> in_shapes=input->tensor_shapes();
        size_t total_input_element=0;
        if(in_shapes.size()==1){
            total_input_element=in_shapes[0];
        }else if(in_shapes.size()==2){
            total_input_element=in_shapes[0]*in_shapes[1];
        }else{
            std::cerr<<"目前只支持一维和二维的输入"<<std::endl;
        }
        if(output->tensor_element_size()!=total_input_element*embed_dims_){
            std::cerr<<"接受输出的Tensor大小不匹配"<<std::endl;
        }
        float *in_ptr=(float *)input->tensor_data_ptr();
        float *out_ptr=(float *)output->tensor_data_ptr();
        float *vocab_ptr=(float *)vocab_->tensor_data_ptr();
        if(in_shapes.size()==1){
            int token_size=in_shapes[0];
            for(int i=0;i<token_size;i++){
                for(int j=0;j<embed_dims_;j++){
                    out_ptr[i*embed_dims_+j]=vocab_ptr[(int)in_ptr[i]*embed_dims_+j];
                }
            }
        }
        if(in_shapes.size()==2){
            int batch_size=in_shapes[0];
            int token_size=in_shapes[1];
            for(int i=0;i<batch_size;i++){
                for(int j=0;j<token_size;j++){
                    for(int k=0;k<embed_dims_;k++){
                        out_ptr[i*token_size*embed_dims_+j*embed_dims_+k]=
                                vocab_ptr[(int)in_ptr[i*token_size+j]*embed_dims_+k];
                    }
                }
            }
        }
    }
};

int main(){
    std::shared_ptr<Allocator> allocator=std::make_shared<CPUAllocator>();
    std::shared_ptr<Tensor> input=std::make_shared<Tensor>(std::vector<int>{2,3},allocator);
    std::shared_ptr<Tensor> output=std::make_shared<Tensor>(std::vector<int>{2,3,5},allocator);
    std::shared_ptr<EmbeddingLayer> emb=std::make_shared<EmbeddingLayer>(10,5,allocator);
    float *in_ptr=(float *)input->tensor_data_ptr();
    for(int i=0;i<input->tensor_element_size();i++){
        in_ptr[i]=i;
    }
    std::cout<<"词嵌入层的词表矩阵"<<std::endl;
    std::shared_ptr<Tensor> vocab=emb->embedding_vocab();
    vocab->tensor_print_data();
    std::cout<<"输入矩阵"<<std::endl;
    input->tensor_print_data();
    emb->forward(input,output);
    std::cout<<"输出矩阵"<<std::endl;
    output->tensor_print_data();
}


