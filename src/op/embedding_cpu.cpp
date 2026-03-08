

namespace hxinfer{
    void embedding_tensor(const std::shared_ptr<Tensor>& token_ids,const std::shared_ptr<Tensor>& weight,
                          std::shared_ptr<Tensor>& output);
}