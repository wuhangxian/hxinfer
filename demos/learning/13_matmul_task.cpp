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

class MatMulLayer:public Layer{
private:
    std::shared_ptr<Tensor> input_b_;
    bool trans_b_;
public:
    MatMulLayer(std::shared_ptr<Tensor> input_b,bool trans_b=true):
            Layer("MatMul"),input_b_(input_b),trans_b_(trans_b){}

    void forward(std::shared_ptr<Tensor> input_a,std::shared_ptr<Tensor> output) override{
        const auto& shape_a=input_a->tensor_shape();
        const auto& shape_b=input_b_->tensor_shape();
        const auto& shape_out=output->tensor_shape();
        int M=shape_a[0];
        int K=shape_a[1];
        int N;
        if(trans_b_){
            if(shape_b[1]!=K){
                std::cerr<<"A和B^T 的内积维度 K 不匹配！"<<'\n';
            }
            N=shape_b[0];
        }else{
            if(shape_b[0]!=K){
                std::cerr<<"A和B的内积维度 K 不匹配！"<<'\n';
            }
            N=shape_b[1];
        }
        if(shape_out[0]!=M||shape_out[1]!=N){
            std::cerr<<"输出 C 的形状必须是 [" << M << ", " << N << "]！"<<'\n';
            return ;
        }
        float *ptr_a=(float *)input_a->tensor_data_ptr();
        float *ptr_b=(float *)input_b_->tensor_data_ptr();
        float *ptr_out=(float *)output->tensor_data_ptr();
        for(int i=0;i<M;i++){
            for(int j=0;j<N;j++){
                float sum=0;
                for(int k=0;k<K;k++){
                    if(trans_b_){
                        sum=sum+ptr_a[i*K+k]*ptr_b[j*K+k];
                    }else{
                        sum=sum+ptr_a[i*K+k]*ptr_b[k*N+j];
                    }
                }
                ptr_out[i*N+j]=sum;
            }
        }
    }
};

int main(){
    std::shared_ptr<Allocator>allocator=std::make_shared<CPUAllocator>();
    std::shared_ptr<Tensor> mat_a=std::make_shared<Tensor>(std::vector<int>{2,3},allocator);
    float *ptr_a=(float *)mat_a->tensor_data_ptr();
    ptr_a[0]=1;ptr_a[1]=2;ptr_a[2]=3;
    ptr_a[3]=4;ptr_a[4]=5;ptr_a[5]=6;
    std::shared_ptr<Tensor> mat_b=std::make_shared<Tensor>(std::vector<int>{2,3},allocator);
    float *ptr_b = (float *)mat_b->tensor_data_ptr();
    ptr_b[0] = 1; ptr_b[1] = 0; ptr_b[2] = 1;
    ptr_b[3] = 0; ptr_b[4] = 1; ptr_b[5] = 0;
    std::shared_ptr<MatMulLayer> matMulLayer=std::make_shared<MatMulLayer>(mat_b);
    std::shared_ptr<Tensor> output=std::make_shared<Tensor>(std::vector<int>{2,2},allocator);
    matMulLayer->forward(mat_a,output);
    std::cout << "--- 矩阵 A [2, 3] ---" << '\n';
    mat_a->tensor_print_data();
    std::cout << "--- 矩阵 B (这里存的是 B 的转置) [2, 3] ---" << '\n';
    mat_b->tensor_print_data();
    std::cout << "--- 结果矩阵 C = A * B^T [2, 2] ---" << '\n';
    output->tensor_print_data();
}
