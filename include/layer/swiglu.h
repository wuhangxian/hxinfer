#ifndef HXINFER_SWIGLU_H
#define HXINFER_SWIGLU_H
#include "layer.h"
#include "linear.h"
namespace hxinfer{
    class SwigluLayer: public Layer{
    private:
        std::shared_ptr<Tensor> w_gate_;
        std::shared_ptr<Tensor> w_up_;
        std::shared_ptr<Tensor> w_down_;

        bool use_fused_ = false;

        std::shared_ptr<LinearLayer> gate_proj_;
        std::shared_ptr<LinearLayer> up_proj_;
        std::shared_ptr<LinearLayer> down_proj_;

        std::shared_ptr<Tensor> after_gate_;
        std::shared_ptr<Tensor> after_up_;

        ModelConfig config_;
    public:
        SwigluLayer(std::shared_ptr<Allocator> allocator,ModelConfig& config,std::shared_ptr<Tensor>& w_gate,
                    std::shared_ptr<Tensor>& w_up,std::shared_ptr<Tensor>& w_down);
        void enable_fused(){ use_fused_ = true; }
        void forward(std::shared_ptr<Tensor>& input,std::shared_ptr<Tensor>& output) override;
    };
}
#endif //HXINFER_SWIGLU_H
