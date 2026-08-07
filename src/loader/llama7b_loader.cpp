#include "loader/llama7b_loader.h"
#include "layer/transformer.h"
#include "layer/embedding.h"
#include "layer/rmsnorm.h"
#include "layer/linear.h"
#include <nlohmann/json.hpp>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace hxinfer{

using json = nlohmann::json;

struct WeightEntry {
    size_t offset;
    size_t nbytes;
    std::vector<int> shape;
};

// 从 mmap 基地址把 FP16 权重直接拷贝到 GPU
static std::shared_ptr<Tensor> load_fp16_gpu(
    const void* base, const WeightEntry& e,
    const std::shared_ptr<CUDAAllocator>& alloc)
{
    auto t = std::make_shared<Tensor>(alloc, e.shape, DataType::kDataTypeFP16);
    t->tensor_set_device_type(DeviceType::kDeviceCUDA);
    cudaMemcpy(t->raw_data_ptr(),
               static_cast<const char*>(base) + e.offset,
               e.nbytes, cudaMemcpyHostToDevice);
    return t;
}

// 从 mmap 基地址读 FP16 权重，在 CPU 转成 FP32 后上传到 GPU
static std::shared_ptr<Tensor> load_fp32_from_fp16_gpu(
    const void* base, const WeightEntry& e,
    const std::shared_ptr<CUDAAllocator>& alloc)
{
    size_t n = e.nbytes / sizeof(uint16_t);
    const uint16_t* src = reinterpret_cast<const uint16_t*>(
        static_cast<const char*>(base) + e.offset);

    std::vector<float> fp32(n);
    for(size_t i=0; i<n; i++){
        fp32[i] = __half2float(*reinterpret_cast<const __half*>(&src[i]));
    }

    auto t = std::make_shared<Tensor>(alloc, e.shape, DataType::kDataTypeFP32);
    t->tensor_set_device_type(DeviceType::kDeviceCUDA);
    cudaMemcpy(t->raw_data_ptr(), fp32.data(),
               n * sizeof(float), cudaMemcpyHostToDevice);
    return t;
}

std::shared_ptr<LlamaModel> Llama7BLoader::load(
    const std::string& data_dir,
    ModelConfig& out_config,
    const std::shared_ptr<CPUAllocator>& /*cpu_alloc*/,
    const std::shared_ptr<CUDAAllocator>& cuda_alloc)
{
    // ── 1. 解析 config.json ──────────────────────────────────────────────
    std::string config_path = data_dir + "/Yarn-Llama-2-7b-128k/config.json";
    std::ifstream cf(config_path);
    if(!cf) throw std::runtime_error("找不到 config.json: " + config_path);
    json cfg = json::parse(cf);

    out_config.dim        = cfg["hidden_size"];
    out_config.hidden_dim = cfg["intermediate_size"];
    out_config.layer      = cfg["num_hidden_layers"];
    out_config.head       = cfg["num_attention_heads"];
    out_config.kv_head    = cfg["num_key_value_heads"];
    out_config.vocab_size = cfg["vocab_size"];
    out_config.seq_len    = 4096;

    // Parse YaRN rope_scaling
    if(cfg.contains("rope_scaling")){
        auto& rs = cfg["rope_scaling"];
        std::string rs_type = rs.value("type", "");
        if(rs_type == "yarn"){
            out_config.rope_use_yarn     = true;
            out_config.rope_factor       = rs.value("factor", 1.0f);
            out_config.rope_orig_max_pos = rs.value("original_max_position_embeddings", 4096);
            out_config.rope_beta_fast    = rs.value("beta_fast", 32.0f);
            out_config.rope_beta_slow    = rs.value("beta_slow", 1.0f);
        }
    }

    std::cout << ">>> [Loader] LLaMA-2 7B 配置\n"
              << "    dim="        << out_config.dim
              << "  hidden_dim="   << out_config.hidden_dim
              << "  layers="       << out_config.layer
              << "  heads="        << out_config.head
              << "  vocab="        << out_config.vocab_size
              << "  seq_len="      << out_config.seq_len;
    if(out_config.rope_use_yarn){
        std::cout << "\n    YaRN: factor=" << out_config.rope_factor
                  << " orig_max_pos=" << out_config.rope_orig_max_pos
                  << " beta_fast=" << out_config.rope_beta_fast
                  << " beta_slow=" << out_config.rope_beta_slow;
    }
    std::cout << "\n";

    // ── 2. 解析 index.json ───────────────────────────────────────────────
    std::string index_path = data_dir + "/weights/llama7b_index.json";
    std::ifstream idxf(index_path);
    if(!idxf) throw std::runtime_error("找不到 index.json: " + index_path);
    json idx = json::parse(idxf);

    std::unordered_map<std::string, WeightEntry> index;
    for(auto& [key, val] : idx.items()){
        WeightEntry e;
        e.offset = val["offset"];
        e.nbytes = val["nbytes"];
        for(auto& d : val["shape"]) e.shape.push_back(d.get<int>());
        index[key] = e;
    }
    std::cout << ">>> [Loader] 索引加载完成，共 " << index.size() << " 个权重\n";

    // ── 3. mmap weights.bin ──────────────────────────────────────────────
    std::string bin_path = data_dir + "/weights/llama7b_weights.bin";
    int fd = open(bin_path.c_str(), O_RDONLY);
    if(fd < 0) throw std::runtime_error("找不到 weights.bin: " + bin_path);
    struct stat sb; fstat(fd, &sb);
    const void* mmap_base = mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if(mmap_base == MAP_FAILED) throw std::runtime_error("mmap 失败");
    close(fd);
    std::cout << ">>> [Loader] mmap " << sb.st_size/1e9 << " GB 权重文件完成\n";

    auto& c = out_config;

    // ── 4. Embedding 权重（FP16）────────────────────────────────────────
    auto embed_w = load_fp16_gpu(mmap_base, index.at("model.embed_tokens.weight"), cuda_alloc);

    // ── 5. 每一层的 TransformerLayer ─────────────────────────────────────
    std::vector<std::shared_ptr<TransformerLayer>> blocks;
    blocks.reserve(c.layer);

    for(int i=0; i<c.layer; i++){
        std::string li = "model.layers." + std::to_string(i);

        auto attn_norm_w = load_fp32_from_fp16_gpu(mmap_base, index.at(li+".input_layernorm.weight"), cuda_alloc);
        auto wq          = load_fp16_gpu(mmap_base, index.at(li+".self_attn.q_proj.weight"),  cuda_alloc);
        auto wk          = load_fp16_gpu(mmap_base, index.at(li+".self_attn.k_proj.weight"),  cuda_alloc);
        auto wv          = load_fp16_gpu(mmap_base, index.at(li+".self_attn.v_proj.weight"),  cuda_alloc);
        auto wo          = load_fp16_gpu(mmap_base, index.at(li+".self_attn.o_proj.weight"),  cuda_alloc);
        auto ffn_norm_w  = load_fp32_from_fp16_gpu(mmap_base, index.at(li+".post_attention_layernorm.weight"), cuda_alloc);
        auto w_gate      = load_fp16_gpu(mmap_base, index.at(li+".mlp.gate_proj.weight"), cuda_alloc);
        auto w_up        = load_fp16_gpu(mmap_base, index.at(li+".mlp.up_proj.weight"),   cuda_alloc);
        auto w_down      = load_fp16_gpu(mmap_base, index.at(li+".mlp.down_proj.weight"), cuda_alloc);

        blocks.push_back(std::make_shared<TransformerLayer>(
            cuda_alloc, c,
            attn_norm_w, wq, wk, wv, wo,
            ffn_norm_w, w_gate, w_up, w_down,
            true /*enable_fused*/));

        if((i+1) % 8 == 0)
            std::cout << "    已加载 " << i+1 << "/" << c.layer << " 层\n";
    }

    // ── 6. 最终 RMSNorm 和 LM Head ───────────────────────────────────────
    auto final_norm_w = load_fp32_from_fp16_gpu(mmap_base, index.at("model.norm.weight"), cuda_alloc);
    auto lm_head_w    = load_fp16_gpu(mmap_base, index.at("lm_head.weight"), cuda_alloc);

    auto final_norm = std::make_shared<RMSNormLayer>(final_norm_w);
    auto lm_head    = std::make_shared<LinearLayer>(lm_head_w);

    // ── 7. 组装 LlamaModel ───────────────────────────────────────────────
    auto embedding_layer = std::make_shared<EmbeddingLayer>(embed_w);

    auto model = std::make_shared<LlamaModel>(
        cuda_alloc,
        embedding_layer, blocks, final_norm, lm_head, out_config);

    munmap(const_cast<void*>(mmap_base), sb.st_size);
    std::cout << ">>> [Loader] 7B 模型加载完成！\n";
    return model;
}

} // namespace hxinfer
