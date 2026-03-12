#include "layer/swiglu.h"
#include "base/allocator.h"
namespace hxinfer{
    SwigluLayer::SwigluLayer(std::shared_ptr<Allocator> allocator,ModelConfig& config,std::shared_ptr<Tensor>& w_gate,
    std::shared_ptr<Tensor>& w_up,std::shared_ptr<Tensor>& w_down): Layer("Swiglu"),
    w_gate_(w_gate),w_up_(w_up),w_down_(w_down), config_(config){
        DataType type_gate=w_gate_->tensor_data_type();
        DataType type_up=w_up_->tensor_data_type();
        int max_seq_len=config_.seq_len;
        int hidden_dim=config_.hidden_dim;
        std::vector<int> max_shapes={max_seq_len,hidden_dim};
        after_gate_=std::make_shared<Tensor>(allocator,max_shapes,type_gate);
        after_up_=std::make_shared<Tensor>(allocator,max_shapes,type_up);
        gate_proj_ = std::make_shared<LinearLayer>(w_gate_);
        up_proj_   = std::make_shared<LinearLayer>(w_up_);
        down_proj_ = std::make_shared<LinearLayer>(w_down_);
    }
    void SwigluLayer::forward(std::shared_ptr<Tensor>& input,std::shared_ptr<Tensor>& output){
    std::vector<int> in_shapes=input->tensor_shapes();
    int dim=config_.dim;
    int hidden_dim=config_.hidden_dim;
    int curr_seq_len=input->tensor_total_elements()/dim;
    std::vector<int> curr_shapes={curr_seq_len,hidden_dim};
    after_gate_->tensor_reshape(curr_shapes);
    after_up_->tensor_reshape(curr_shapes);
// 🚀 3. 使用正确的成员变量名
    gate_proj_->forward(input, after_gate_);
    silu_tensor(after_gate_, after_gate_);

    up_proj_->forward(input, after_up_);
    mul_tensor(after_gate_, after_up_, after_gate_);

    down_proj_->forward(after_gate_, output);
}
}