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

        std::shared_ptr<Tensor> after_qkt_;
        std::shared_ptr<Tensor> after_qktv_;
        ModelConfig config_;
    public:
        AttentionLayer(std::shared_ptr<Allocator> allocator,ModelConfig& config,std::shared_ptr<Tensor>& wq,
                       std::shared_ptr<Tensor>& wk,std::shared_ptr<Tensor>& wv,std::shared_ptr<Tensor>& wo):
                Layer("Attention"), config_(config),wq_(wq),wk_(wk),wv_(wv),wo_(wo){
            int dim=config_.dim;
            int max_seq_len=config_.seq_len;
            int head=config_.head;
            DataType dtype=wq_->tensor_data_type();
            std::vector<int> curr_shapes={1,dim};
            curr_q_=std::make_shared<Tensor>(allocator,curr_shapes,dtype);
            curr_k_=std::make_shared<Tensor>(allocator,curr_shapes,dtype);
            curr_v_=std::make_shared<Tensor>(allocator,curr_shapes,dtype);
            // 显存池创建,构造时一次性分配
            // 这里 max_seq_len = config_.seq_len = 256，dim = 288
            std::vector<int> cache_shapes={max_seq_len,dim};
            k_cache_=std::make_shared<Tensor>(allocator,cache_shapes,dtype);
            v_cache_=std::make_shared<Tensor>(allocator,cache_shapes,dtype);

            std::vector<int> score_shapes={head,max_seq_len,max_seq_len};
            after_qkt_=std::make_shared<Tensor>(allocator,score_shapes,dtype);
            after_qktv_=std::make_shared<Tensor>(allocator,curr_shapes,dtype);

            q_proj_=std::make_shared<LinearLayer>(wq_);
            k_proj_=std::make_shared<LinearLayer>(wk_);
            v_proj_=std::make_shared<LinearLayer>(wv_);
            o_proj_=std::make_shared<LinearLayer>(wo_);

        }

        void forward(std::shared_ptr<Tensor>& input,std::shared_ptr<Tensor>& output,int pos);

        // ===================== Prefix Cache 专用接口 =====================
        // 为什么要暴露 KV Cache？
        // PrefixCacheManager 需要：
        //   1. store(): 从 k_cache_ / v_cache_ 中 memcpy 出前 N 行做快照
        //   2. restore(): 把快照 memcpy 回 k_cache_ / v_cache_ 的前 N 行
        // 返回引用是为了让外部能直接拿到底层指针做 memcpy，不产生拷贝
        std::shared_ptr<Tensor>& get_k_cache() { return k_cache_; }
        std::shared_ptr<Tensor>& get_v_cache() { return v_cache_; }

        // 🚀 补上这份"基础合同"，红线瞬间灰飞烟灭！
        void forward(std::shared_ptr<Tensor>& input, std::shared_ptr<Tensor>& output) override {
            // 因为我们强制要求带 pos，如果有人调错版本，直接报错
            throw std::runtime_error("AttentionLayer requires 'pos' parameter!\n");
        }
    };
}


#endif //HXINFER_ATTENTION_H
