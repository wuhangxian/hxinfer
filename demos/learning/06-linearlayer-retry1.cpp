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
    void *allocate(size_t byte_size) override{
        void *ptr=malloc(byte_size);
        if(ptr== nullptr) {
            return nullptr;
        }
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
public:
    Tensor(std::vector<int> shapes,std::shared_ptr<Allocator> allocator):
                shapes_(shapes){
        int total_elements=std::accumulate(shapes_.begin(),shapes_.end(),1,std::multiplies<>());
        size_t byte_size=total_elements*sizeof (float );
        buffer_=std::make_shared<Buffer>(byte_size,allocator);
    }
    const std::vector<int> &tensor_shapes(){
        return shapes_;
    }
    const void* tensor_data_ptr() const{
        return buffer_->buffer_data_ptr();
    }
    void* tensor_data_ptr(){
        return buffer_->buffer_data_ptr();
    }
    int tensor_element_size() const{
        return std::accumulate(shapes_.begin(),shapes_.end(),1,std::multiplies<>());
    }
    float &tensor_at(int index){
        float *p=(float *)tensor_data_ptr();
        return p[index];
    }
    void tensor_print_data(){
        float *p=(float *)tensor_data_ptr();
        int n=shapes_.size();
        if(n==0||buffer_== nullptr){
            std::cerr<<"该tensor中无数据"<<std::endl;
        }
        if(n==1){
            std::cout<<"[";
            for(int i=0;i<shapes_[0];i++){
                std::cout<<p[i]<<" ";
            }
            std::cout<<"]"<<std::endl;
        }
        if(n==2){
            int m=shapes_[0];
            int k=shapes_[1];
            std::cout<<"["<<std::endl;
            for(int i=0;i<m;i++){
                std::cout<<"[";
                for(int j=0;j<k;j++){
                    std::cout<<p[i*k+j]<<" ";
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
            Layer("Linear"),in_features_(in_features),out_features_(out_features),has_bias_(has_bias){
        weights_=std::make_shared<Tensor>(std::vector<int>{in_features_,out_features_},allocator);
        float *w_ptr=(float *)weights_->tensor_data_ptr();
        for(int  i=0; i<weights_->tensor_element_size();i++){
            w_ptr[i]=1.0f;
        }
        if(has_bias_){
            bias_=std::make_shared<Tensor>(std::vector<int>{out_features_},allocator);
            float *b_ptr=(float *)bias_->tensor_data_ptr();
            for(int i=0;i<bias_->tensor_element_size();i++){
                b_ptr[i]=1.0f;
            }
        }else{
            bias_= nullptr;
        }
    }

    void forward(std::shared_ptr<Tensor> input,std::shared_ptr<Tensor> output) override{
        const std::vector<int>& in_shapes=input->tensor_shapes();
        if(in_shapes.size()!=2){
            std::cerr<<"Linear only support 2D input for now!"<<std::endl;
            return;
        }
        int batch=in_shapes[0];
        int K=in_shapes[1];
        int N=out_features_;
        if(K!=in_features_){
            std::cerr<<"Input features mismatch"<<std::endl;
            return;
        }
        float *in_ptr=(float *)input->tensor_data_ptr();
        float *out_ptr=(float *)output->tensor_data_ptr();
        float *w_ptr=(float *)weights_->tensor_data_ptr();
        float *b_ptr=has_bias_?(float *)bias_->tensor_data_ptr(): nullptr;
        for(int i=0;i<batch;i++){
            for(int j=0;j<N;j++){
                float sum=0.0f;
                for(int k=0;k<K;k++){
                    sum=sum+in_ptr[i*K+k]*w_ptr[k*N+j];
                }
                if(has_bias_){
                    sum=sum+b_ptr[j];
                }
                out_ptr[i*N+j]=sum;
            }
        }

    }
};
int main(){
    std::shared_ptr<Allocator> allocator=std::make_shared<CPUAllocator>();
    std::shared_ptr<Tensor> input = std::make_shared<Tensor>(std::vector<int>{2, 3}, allocator);
    float* in_p = (float *)input->tensor_data_ptr();
    in_p[0]=1; in_p[1]=1; in_p[2]=1;
    in_p[3]=2; in_p[4]=2; in_p[5]=2;
    std::cout << "--- Input ---" << std::endl;
    input->tensor_print_data();
    std::shared_ptr<Tensor> output = std::make_shared<Tensor>(std::vector<int>{2, 2}, allocator);
    std::cout << "\n--- Test 1: With Bias (Expect 4 and 7) ---" << std::endl;
    LinearLayer linear_with_bias(3, 2, true, allocator);
    linear_with_bias.forward(input, output);
    output->tensor_print_data();
    // 解释: [1,1,1] * [1,1,1]^T = 3, + bias(1) = 4
    // 解释: [2,2,2] * [1,1,1]^T = 6, + bias(1) = 7

    // 【测试 2】不带 Bias 的 Linear
    std::cout << "\n--- Test 2: No Bias (Expect 3 and 6) ---" << std::endl;
    LinearLayer linear_no_bias(3, 2, false, allocator);
    linear_no_bias.forward(input, output);
    output->tensor_print_data();
    // 解释: [1,1,1] * [1,1,1]^T = 3, 不加bias = 3
    // 解释: [2,2,2] * [1,1,1]^T = 6, 不加bias = 6
    return 0;
}