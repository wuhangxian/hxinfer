#ifndef HXINFER_ATTENTION_H
#define HXINFER_ATTENTION_H
#include "layer.h"
#include "linear.h"
namespace hxinfer{
    class AttentionLayer:public Layer{
    private:
        std::shared_ptr<Tensor> wq_;
        std::shared_ptr<Tensor> wk_;
        std::shared_ptr<Tensor> wv_;
        std::shared_ptr<Tensor> wo_;

        std::shared_ptr<LinearLayer> q_proj_;
        std::shared_ptr<LinearLayer> k_proj_;
        std::shared_ptr<LinearLayer> v_proj_;
        std::shared_ptr<LinearLayer> o_proj_;

        std::shared_ptr<Tensor> curr_q_;
        std::shared_ptr<Tensor> curr_k_;
        std::shared_ptr<Tensor> curr_v_;

        std::shared_ptr<Tensor> k_cache_;
        std::shared_ptr<Tensor> v_cache_;

        std::shared_ptr<Tensor> after_qktv_;
        ModelConfig config_;
        DataType activation_dtype_;

        bool has_bias_ = false;
        std::shared_ptr<Tensor> bq_;
        std::shared_ptr<Tensor> bk_;
        std::shared_ptr<Tensor> bv_;

    public:
        AttentionLayer(std::shared_ptr<Allocator> allocator, ModelConfig& config,
                       std::shared_ptr<Tensor>& wq, std::shared_ptr<Tensor>& wk,
                       std::shared_ptr<Tensor>& wv, std::shared_ptr<Tensor>& wo):
                Layer("Attention"), config_(config), wq_(wq), wk_(wk), wv_(wv), wo_(wo),
                has_bias_(false){
            int dim        = config_.dim;
            int max_seq_len = config_.seq_len;
            activation_dtype_ = (wq_->tensor_data_type() == DataType::kDataTypeFP16)
                                ? DataType::kDataTypeFP16 : DataType::kDataTypeFP32;
            int kv_dim = config_.kv_head * (config_.dim / config_.head);
            std::vector<int> q_shape = {max_seq_len, dim};
            std::vector<int> kv_shape = {max_seq_len, kv_dim};
            curr_q_ = std::make_shared<Tensor>(allocator, q_shape, activation_dtype_);
            curr_k_ = std::make_shared<Tensor>(allocator, kv_shape, activation_dtype_);
            curr_v_ = std::make_shared<Tensor>(allocator, kv_shape, activation_dtype_);
            std::vector<int> cache_shapes = {max_seq_len, kv_dim};
            k_cache_ = std::make_shared<Tensor>(allocator, cache_shapes, activation_dtype_);
            v_cache_ = std::make_shared<Tensor>(allocator, cache_shapes, activation_dtype_);
            after_qktv_ = std::make_shared<Tensor>(allocator, q_shape, activation_dtype_);
            DeviceType dev = allocator->device_type();
            curr_q_->tensor_set_device_type(dev);
            curr_k_->tensor_set_device_type(dev);
            curr_v_->tensor_set_device_type(dev);
            k_cache_->tensor_set_device_type(dev);
            v_cache_->tensor_set_device_type(dev);
            after_qktv_->tensor_set_device_type(dev);
            q_proj_ = std::make_shared<LinearLayer>(wq_);
            k_proj_ = std::make_shared<LinearLayer>(wk_);
            v_proj_ = std::make_shared<LinearLayer>(wv_);
            o_proj_ = std::make_shared<LinearLayer>(wo_);
        }

        AttentionLayer(std::shared_ptr<Allocator> allocator, ModelConfig& config,
                       std::shared_ptr<Tensor>& wq, std::shared_ptr<Tensor>& wk,
                       std::shared_ptr<Tensor>& wv, std::shared_ptr<Tensor>& wo,
                       std::shared_ptr<Tensor>& bq, std::shared_ptr<Tensor>& bk,
                       std::shared_ptr<Tensor>& bv):
                Layer("Attention"), config_(config), wq_(wq), wk_(wk), wv_(wv), wo_(wo),
                has_bias_(true), bq_(bq), bk_(bk), bv_(bv){
            int dim        = config_.dim;
            int max_seq_len = config_.seq_len;
            activation_dtype_ = (wq_->tensor_data_type() == DataType::kDataTypeFP16)
                                ? DataType::kDataTypeFP16 : DataType::kDataTypeFP32;
            int kv_dim = config_.kv_head * (config_.dim / config_.head);
            std::vector<int> q_shape = {max_seq_len, dim};
            std::vector<int> kv_shape = {max_seq_len, kv_dim};
            curr_q_ = std::make_shared<Tensor>(allocator, q_shape, activation_dtype_);
            curr_k_ = std::make_shared<Tensor>(allocator, kv_shape, activation_dtype_);
            curr_v_ = std::make_shared<Tensor>(allocator, kv_shape, activation_dtype_);
            std::vector<int> cache_shapes = {max_seq_len, kv_dim};
            k_cache_ = std::make_shared<Tensor>(allocator, cache_shapes, activation_dtype_);
            v_cache_ = std::make_shared<Tensor>(allocator, cache_shapes, activation_dtype_);
            after_qktv_ = std::make_shared<Tensor>(allocator, q_shape, activation_dtype_);
            DeviceType dev = allocator->device_type();
            curr_q_->tensor_set_device_type(dev);
            curr_k_->tensor_set_device_type(dev);
            curr_v_->tensor_set_device_type(dev);
            k_cache_->tensor_set_device_type(dev);
            v_cache_->tensor_set_device_type(dev);
            after_qktv_->tensor_set_device_type(dev);
            q_proj_ = std::make_shared<LinearLayer>(wq_);
            k_proj_ = std::make_shared<LinearLayer>(wk_);
            v_proj_ = std::make_shared<LinearLayer>(wv_);
            o_proj_ = std::make_shared<LinearLayer>(wo_);
        }

        void forward(std::shared_ptr<Tensor>& input, std::shared_ptr<Tensor>& output, int pos);
        void forward_prefill(std::shared_ptr<Tensor>& input, std::shared_ptr<Tensor>& output, int seq_len);

        std::shared_ptr<Tensor>& get_k_cache() { return k_cache_; }
        std::shared_ptr<Tensor>& get_v_cache() { return v_cache_; }

        void forward(std::shared_ptr<Tensor>& input, std::shared_ptr<Tensor>& output) override {
            throw std::runtime_error("AttentionLayer requires 'pos' parameter!\n");
        }
    };
}
#endif
