#include "loader/llama15m_loader.h"
#include "base/config.h"
#include <fcntl.h>    // 控制文件操作（定义了 O_RDONLY 等宏）
#include <sys/mman.h>
#include <unistd.h>   // 提供系统调用接口（如 open, close, read）
#include <sys/types.h> // 定义了一些系统级的数据类型（虽然 open 有时不强制，但建议带上）
#include <sys/stat.h>
#include "iostream"
#include "memory"

namespace hxinfer{
    std::shared_ptr<LlamaModel> Llama15MLoader::load_model(const std::string &model_path,
                                                           hxinfer::ModelConfig &out_config,
                                                           const std::shared_ptr<Allocator> allocator) {
        // ==========================================
        // 1. 物理层：mmap 零拷贝映射
        // ==========================================
        int fd=open(model_path.c_str(),O_RDONLY);
        if (fd < 0) throw std::runtime_error("找不到模型文件: " + model_path);
        struct stat sb;
        fstat(fd, &sb);
        float* data = static_cast<float*>(mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
        if (data == MAP_FAILED) throw std::runtime_error("mmap 失败");

        // 读取头部的 Config
        ModelConfig* file_cfg = reinterpret_cast<ModelConfig*>(data);
        out_config = *file_cfg;
        std::cout << ">>> [Loader] 命中 LLaMA 15M 格式！"<<'\n'
                    <<'\n'<<"模型维度: " << out_config.dim
                    <<'\n'<<"模型隐藏层维度: " << out_config.hidden_dim
                    <<'\n'<<"模型层数: " << out_config.layer
                    <<'\n'<<"模型头数: " << out_config.head
                    <<'\n'<<"KV头数: " << out_config.kv_head
                    <<'\n'<<"词表大小:"<<out_config.vocab_size
                    <<'\n'<<"模型支持最大长度:"<<out_config.seq_len
                    <<std::endl;

        // 指针跳过头部，指向第一个权重
        float* w_ptr = data + (sizeof(ModelConfig) / sizeof(float));

        // 提前算好一些常量
        int head_dim = out_config.dim / out_config.head;
        int kv_dim = (out_config.kv_head) * head_dim;

        // ==========================================
        // 2. 内存申请与搬运：按文件排布顺序疯狂吸入！
        // ==========================================
        // 2.1 Embedding 权重
        auto embed_w = std::make_shared<Tensor>(allocator, std::vector<int>{out_config.vocab_size, out_config.dim}, DataType::kDataTypeFP32);
        std::memcpy(embed_w->raw_data_ptr(), w_ptr, out_config.vocab_size * out_config.dim * sizeof(float));
        w_ptr += out_config.vocab_size * out_config.dim;
        // 2.2 准备所有层的 Tensor 容器
        std::vector<std::shared_ptr<Tensor>> attn_norm_w(out_config.layer);
        std::vector<std::shared_ptr<Tensor>> wq_w(out_config.layer), wk_w(out_config.layer), wv_w(out_config.layer), wo_w(out_config.layer);
        std::vector<std::shared_ptr<Tensor>> ffn_norm_w(out_config.layer);
        std::vector<std::shared_ptr<Tensor>> w1_w(out_config.layer), w2_w(out_config.layer), w3_w(out_config.layer);

        auto copy_tensor = [&](std::shared_ptr<Tensor>& t, const std::vector<int>& shape) {
            t = std::make_shared<Tensor>(allocator, shape, DataType::kDataTypeFP32);
            size_t elements = 1;
            for(int s : shape) elements *= s;
            std::memcpy(t->raw_data_ptr(), w_ptr, elements * sizeof(float));
            w_ptr += elements;
        };
        for (int i = 0; i < out_config.layer; i++) copy_tensor(attn_norm_w[i], {out_config.dim});
        for (int i = 0; i < out_config.layer; i++) copy_tensor(wq_w[i], {out_config.dim, out_config.dim});
        for (int i = 0; i < out_config.layer; i++) copy_tensor(wk_w[i], {kv_dim, out_config.dim});
        for (int i = 0; i < out_config.layer; i++) copy_tensor(wv_w[i], {kv_dim, out_config.dim});
        for (int i = 0; i < out_config.layer; i++) copy_tensor(wo_w[i], {out_config.dim, out_config.dim});
        for (int i = 0; i < out_config.layer; i++) copy_tensor(ffn_norm_w[i], {out_config.dim});
        for (int i = 0; i < out_config.layer; i++) copy_tensor(w1_w[i], {out_config.hidden_dim, out_config.dim}); // Gate
        for (int i = 0; i < out_config.layer; i++) copy_tensor(w2_w[i], {out_config.dim, out_config.hidden_dim}); // Down
        for (int i = 0; i < out_config.layer; i++) copy_tensor(w3_w[i], {out_config.hidden_dim, out_config.dim}); // Up

        // 2.3 Final Norm
        auto final_norm_w = std::make_shared<Tensor>(allocator, std::vector<int>{out_config.dim}, DataType::kDataTypeFP32);
        std::memcpy(final_norm_w->raw_data_ptr(), w_ptr, out_config.dim * sizeof(float));
        w_ptr += out_config.dim;

        // 2.4 跳过预计算的 RoPE 频率参数 (我们是动态计算的，不需要存)
        w_ptr += out_config.seq_len * head_dim;

        // 2.5 LM_Head 权重 (极其优雅的权重共享拦截)
        auto lm_head_w = std::make_shared<Tensor>(allocator, std::vector<int>{out_config.vocab_size, out_config.dim}, DataType::kDataTypeFP32);
        // 如果文件读到末尾了 (EOF)，说明命中权重共享！
        size_t current_offset = (char*)w_ptr - (char*)data;
        if (current_offset >= sb.st_size) {
            std::cout << ">>> [Loader] 命中权重共享机制！复用 Embedding 内存。" << std::endl;
            std::memcpy(lm_head_w->raw_data_ptr(), embed_w->raw_data_ptr(), out_config.vocab_size * out_config.dim * sizeof(float));
        } else {
            std::memcpy(lm_head_w->raw_data_ptr(), w_ptr, out_config.vocab_size * out_config.dim * sizeof(float));
        }

        // 释放物理内存映射
        munmap(data, sb.st_size);
        close(fd);

        // ==========================================
        // 3. 高达组装：把零散的 Tensor 实例化成 Layer
        // ==========================================
        auto embedding_layer = std::make_shared<EmbeddingLayer>(embed_w);
        // 如果你需要把 weight 塞进 embedding_layer，需要修改 EmbeddingLayer 让他能接收 embed_w

        auto final_norm_layer = std::make_shared<RMSNormLayer>(final_norm_w); // 假设你的类长这样
        auto lm_head_layer = std::make_shared<LinearLayer>(lm_head_w);        // 假设你的类长这样

        std::vector<std::shared_ptr<TransformerLayer>> blocks;
        for (int i = 0; i < out_config.layer; i++) {
            // 这里调用你的 TransformerLayer 构造函数！
            // 把对应的 attn_norm_w[i], wq_w[i]... 传进去！
            blocks.push_back(std::make_shared<TransformerLayer>(
                    allocator, out_config,
                    attn_norm_w[i], wq_w[i], wk_w[i], wv_w[i], wo_w[i],
                    ffn_norm_w[i], w1_w[i], w3_w[i], w2_w[i] // 注意 w1=gate, w3=up, w2=down 的顺序！
            ));
        }

        std::cout << ">>> [Loader] LlamaModel 内部组件实例化完毕，准备交付！" << std::endl;

        return std::make_shared<LlamaModel>(
                allocator, embedding_layer, blocks, final_norm_layer, lm_head_layer, out_config
        );


    }
}