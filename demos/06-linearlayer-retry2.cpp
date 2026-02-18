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
    void *allocate(size_t byte_size) override{
        void *ptr=malloc(byte_size);
        if(ptr== nullptr){
            return nullptr;
        }
        memset(ptr,0,byte_size);
        return ptr;
    }
    void release(void* ptr) override{
        if(ptr!=nullptr){
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
    size_t buffer_byte_size () const{
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
        size_t total_elements=std::accumulate(shapes_.begin(),shapes_.end()
                                ,1,std::multiplies<>());
        size_t byte_size=total_elements*sizeof (float );
        buffer_=std::make_shared<Buffer>(byte_size,allocator);
    }
    const std::vector<int> &tensor_shape() const{
        return shapes_;
    }
    void* tensor_data_ptr(){
        if(buffer_== nullptr){
            return nullptr;
        }
        return buffer_->buffer_data_ptr();
    }
    const void* tensor_data_ptr() const{
        if(buffer_== nullptr){
            return nullptr;
        }
        return buffer_->buffer_data_ptr();
    }
    const size_t tensor_element_size() const{
        return std::accumulate(shapes_.begin(),shapes_.end()
                ,1,std::multiplies<>());
    }
    const void tensor_print_data() const{
         int n=shapes_.size();
         float *p=(float *)tensor_data_ptr();
         if(n==0||buffer_== nullptr) return;
         if(n==1){
             std::cout<<"[";
             for(int i=0;i<shapes_[0];i++) {
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
    }
};

class Layer{
public:
    std::string layer_name_;
    Layer(std::string layer_name):layer_name_(layer_name){}
    virtual void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output)=0;
    virtual ~Layer(){};
};

class LinearLayer:public Layer{
private:
    std::shared_ptr<Tensor> weights_;
    std::shared_ptr<Tensor> bias_;
    int in_features_;
    int out_features_;
    bool has_bias_;
public:
    LinearLayer(int in_features,int out_features,bool has_bias,std::shared_ptr<Allocator> allocator):
            Layer("Linear"),in_features_(in_features)
            ,out_features_(out_features),has_bias_(has_bias){
        weights_=std::make_shared<Tensor>
                    (std::vector<int>{in_features_,out_features_},allocator);
        float *w_ptr=(float *)weights_->tensor_data_ptr();
        size_t w_elemnt_size=weights_->tensor_element_size();
        for(size_t i=0;i<w_elemnt_size;i++){
            w_ptr[i]=1.0f;
        }
        std::cout<<"权重矩阵如下"<<std::endl;
        weights_->tensor_print_data();
        bias_=std::make_shared<Tensor>(std::vector<int>{out_features_},allocator);
        if(has_bias_){
            float *b_ptr=(float *)bias_->tensor_data_ptr();
            size_t b_element_size=bias_->tensor_element_size();
            for(size_t i=0;i<b_element_size;i++){
                b_ptr[i]=1.0f;
            }
            std::cout<<"偏置矩阵如下"<<std::endl;
            bias_->tensor_print_data();
        }else{
            std::cout<<"无偏置矩阵"<<std::endl;
            bias_= nullptr;
        }
    }
    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override{
        const std::vector<int> in_shapes=input->tensor_shape();
        if(in_shapes.size()!=2){
            std::cerr<<"输入的Tensor格式不对"<<std::endl;
            return;
        }
        float *in_ptr=(float *)input->tensor_data_ptr();
        float *w_ptr=(float *)weights_->tensor_data_ptr();
        float *b_ptr = nullptr;
        if(has_bias_ && bias_ != nullptr){
            b_ptr=(float *)bias_->tensor_data_ptr();
        }
        float *out_ptr=(float *)output->tensor_data_ptr();
        int batch=in_shapes[0];
        for(int i=0;i<batch;i++){
            for(int j=0;j<out_features_;j++){
                float sum=0.0f;
                for(int k=0;k<in_features_;k++){
                    sum=sum+in_ptr[i*in_features_+k]*w_ptr[k*out_features_+j];
                }
                if(has_bias_){
                    sum=sum+b_ptr[j];
                }
                out_ptr[i*out_features_+j]=sum;
            }
        }
    }
};
int main(){
    std::shared_ptr<Allocator> allocator=std::make_shared<CPUAllocatr>();
    std::shared_ptr<Tensor> input=std::make_shared<Tensor>(std::vector<int>{2,3},allocator);
    std::shared_ptr<Tensor> output=std::make_shared<Tensor>(std::vector<int>{2,2},allocator);
    float *in_ptr=(float *)input->tensor_data_ptr();
    in_ptr[0]=1.0;
    in_ptr[1]=1.0;
    in_ptr[2]=1.0;
    in_ptr[3]=1.0;
    in_ptr[4]=1.0;
    in_ptr[5]=1.0;
    std::shared_ptr<Layer> linearlayer1=std::make_shared<LinearLayer>
                (3,2, true,allocator);
    linearlayer1->forward(input,output);
    std::cout<<"有偏置矩阵结果"<<std::endl;
    output->tensor_print_data();
    std::shared_ptr<Layer> linearlayer2=std::make_shared<LinearLayer>
            (3,2, false,allocator);
    linearlayer2->forward(input,output);
    std::cout<<"无偏置矩阵结果"<<std::endl;
    output->tensor_print_data();
}


