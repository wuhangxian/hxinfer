#ifndef HXINFER_PREFIX_CACHE_H
#define HXINFER_PREFIX_CACHE_H

#include <unordered_map>
#include <vector>
#include <memory>
#include <cstring>
#include <iostream>
#include "base/allocator.h"
#include "base/config.h"
#include "tensor/tensor.h"
#include "model/llama_model.h"

namespace hxinfer {

    class PrefixCacheManager {
    private:

        struct CachedKVBlock {
            std::vector<int> token_ids;   // 这段前缀的完整 token 序列
            int prefix_len;               // = token_ids.size()

            // layer_kv[i] = { K快照, V快照 } 对应模型第 i 层
            // K快照形状: [prefix_len, dim]，V快照形状: [prefix_len, dim]
            std::vector<std::pair<
                std::shared_ptr<Tensor>,  // K cache snapshot V cache snapshot
                std::shared_ptr<Tensor> >> layer_kv;
        };

        // cache_map_ — 核心缓存哈希表
        // key:   token 序列的 hash 值（size_t）
        // value: 对应的 CachedKVBlock（包含完整的 KV 快照）
        std::unordered_map<size_t, std::shared_ptr<CachedKVBlock>> cache_map_;
        //vector<int> 的比较是 O(N) 的，hash 查找是 O(1) 的。

        // allocator_ — 用于分配快照 Tensor 的内存分配器
        std::shared_ptr<Allocator> allocator_;

        ModelConfig config_;

        //计算哈希值
        size_t compute_hash(const std::vector<int>& tokens) const {
            size_t hash = 0xcbf29ce484222325ULL; // FNV offset basis (64-bit)
            for (int tok : tokens) {
                hash ^= static_cast<size_t>(tok);
                hash *= 0x100000001b3ULL;         // FNV prime (64-bit)
            }
            return hash;
        }

    public:


        PrefixCacheManager(std::shared_ptr<Allocator> allocator, ModelConfig config)
            : allocator_(allocator), config_(config) {}


        void store(const std::vector<int>& prefix_tokens, LlamaModel& model) {
            size_t hash = compute_hash(prefix_tokens);
            int prefix_len = static_cast<int>(prefix_tokens.size());
            int dim = config_.dim;
            int num_layers = config_.layer;

            auto block = std::make_shared<CachedKVBlock>();
            block->token_ids = prefix_tokens;
            block->prefix_len = prefix_len;
            block->layer_kv.resize(num_layers);

            // 获取模型所有 TransformerLayer 的引用
            auto& blocks = model.get_blocks();

            for (int i = 0; i < num_layers; i++) {
                // 穿透调用链: blocks[i] → get_attention() → get_k/v_cache()
                auto& attn = blocks[i]->get_attention();
                auto& k_cache = attn->get_k_cache(); // 形状 [256, 288]
                auto& v_cache = attn->get_v_cache(); // 形状 [256, 288]

                // 分配快照 Tensor：只存前 prefix_len 行
                // 形状 [prefix_len, dim]，比完整 cache [256, dim] 小得多
                auto k_snapshot = std::make_shared<Tensor>(
                    allocator_, std::vector<int>{prefix_len, dim}, DataType::kDataTypeFP32);
                auto v_snapshot = std::make_shared<Tensor>(
                    allocator_, std::vector<int>{prefix_len, dim}, DataType::kDataTypeFP32);

                // *** 核心操作：memcpy **
                // 我们只需要前 prefix_len 行，即前 prefix_len * dim 个 float
                size_t copy_bytes = static_cast<size_t>(prefix_len) * dim * sizeof(float);
                std::memcpy(k_snapshot->raw_data_ptr(), k_cache->raw_data_ptr(), copy_bytes);
                std::memcpy(v_snapshot->raw_data_ptr(), v_cache->raw_data_ptr(), copy_bytes);

                block->layer_kv[i] = {k_snapshot, v_snapshot};
            }

            // 存入哈希表
            cache_map_[hash] = block;

            std::cout << "[PrefixCache] 已缓存 prefix（" << prefix_len
                      << " tokens, " << num_layers << " 层, hash="
                      << hash << "）" << std::endl;
        }


        int lookup(const std::vector<int>& prefix_tokens) const {
            size_t hash = compute_hash(prefix_tokens);
            auto it = cache_map_.find(hash);
            if (it == cache_map_.end()) {
                return 0; 
            }

            // hash 命中了，但可能是碰撞，精确验证 token 序列
            const auto& block = it->second;
            if (block->token_ids == prefix_tokens) {
                return block->prefix_len; // 确实命中
            }

            return 0; // hash 碰撞，实际 token 不同
        }


        void restore(const std::vector<int>& prefix_tokens, LlamaModel& model) {
            size_t hash = compute_hash(prefix_tokens);
            auto it = cache_map_.find(hash);
            if (it == cache_map_.end()) {
                std::cerr << "[PrefixCache] restore 失败：找不到缓存！" << std::endl;
                return;
            }

            const auto& block = it->second;
            int prefix_len = block->prefix_len;
            int dim = config_.dim;
            int num_layers = config_.layer;

            auto& blocks = model.get_blocks();

            for (int i = 0; i < num_layers; i++) {
                auto& attn = blocks[i]->get_attention();
                auto& k_cache = attn->get_k_cache();
                auto& v_cache = attn->get_v_cache();

                const auto& [k_snapshot, v_snapshot] = block->layer_kv[i];

                // *** 核心操作：反向 memcpy ***
                // 把快照的内容拷贝回模型的 KV Cache 的前 prefix_len 行
                size_t copy_bytes = static_cast<size_t>(prefix_len) * dim * sizeof(float);
                std::memcpy(k_cache->raw_data_ptr(), k_snapshot->raw_data_ptr(), copy_bytes);
                std::memcpy(v_cache->raw_data_ptr(), v_snapshot->raw_data_ptr(), copy_bytes);
            }

            std::cout << "[PrefixCache] 已恢复 prefix（" << prefix_len
                      << " tokens → 跳过 " << prefix_len << " 步 forward）" << std::endl;
        }

        // 查看当前缓存了多少条 prefix
        size_t cache_size() const { return cache_map_.size(); }
    };

} // namespace hxinfer

#endif // HXINFER_PREFIX_CACHE_H
