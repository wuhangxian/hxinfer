#include "qwen_weight_loader.h"
#include "layer/transformer.h"
#include "layer/embedding.h"
#include "layer/rmsnorm.h"
#include "layer/linear.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace hxinfer {

using json = nlohmann::json;

static std::string weight_name(int layer, const std::string& suffix) {
    return "model.layers." + std::to_string(layer) + "." + suffix;
}

std::shared_ptr<Qwen2Model> QwenWeightLoader::load(
    const std::string& model_dir,
    ModelConfig& out_config,
    const std::shared_ptr<CPUAllocator>& /*cpu_alloc*/,
    const std::shared_ptr<CUDAAllocator>& cuda_alloc)
{
    std::string config_path = model_dir + "/config.json";
    std::ifstream cf(config_path);
    if (!cf) throw std::runtime_error("config.json not found: " + config_path);
    json cfg = json::parse(cf);

    out_config.dim        = cfg["hidden_size"].get<int>();
    out_config.hidden_dim = cfg["intermediate_size"].get<int>();
    out_config.layer      = cfg["num_hidden_layers"].get<int>();
    out_config.head       = cfg["num_attention_heads"].get<int>();
    out_config.kv_head    = cfg.value("num_key_value_heads", out_config.head);
    out_config.vocab_size = cfg["vocab_size"].get<int>();
    out_config.seq_len    = 4096;
    out_config.rope_theta = cfg.value("rope_theta", 1000000.0f);

    if (cfg.contains("rope_scaling")) {
        auto& rs = cfg["rope_scaling"];
        std::string rs_type = rs.value("type", rs.value("rope_type", ""));
        if (rs_type == "yarn") {
            out_config.rope_use_yarn     = true;
            out_config.rope_factor       = rs.value("factor", 1.0f);
            out_config.rope_orig_max_pos = rs.value("original_max_position_embeddings", 4096);
            out_config.rope_beta_fast    = rs.value("beta_fast", 32.0f);
            out_config.rope_beta_slow    = rs.value("beta_slow", 1.0f);
        }
    }

    std::cout << ">>> [Loader] Qwen2 Model config\n"
              << "    dim=" << out_config.dim
              << "  hidden_dim=" << out_config.hidden_dim
              << "  layers=" << out_config.layer
              << "  heads=" << out_config.head
              << "  kv_heads=" << out_config.kv_head
              << "  vocab=" << out_config.vocab_size
              << "  seq_len=" << out_config.seq_len
              << "  rope_theta=" << out_config.rope_theta << "\n";

    SafetensorsMultiReader reader;
    if (!reader.load_directory(model_dir)) {
        throw std::runtime_error("No .safetensors files found in: " + model_dir);
    }
    std::cout << ">>> [Loader] Safetensors loaded, " << reader.num_tensors() << " tensors\n";

    auto& c = out_config;

    auto embed_w = reader.get_tensor_gpu("model.embed_tokens.weight", cuda_alloc, DataType::kDataTypeFP16);

    std::vector<std::shared_ptr<TransformerLayer>> blocks;
    blocks.reserve(c.layer);

    for (int i = 0; i < c.layer; i++) {
        auto attn_norm_w = reader.get_tensor_gpu(
            weight_name(i, "input_layernorm.weight"), cuda_alloc, DataType::kDataTypeFP32);
        auto ffn_norm_w = reader.get_tensor_gpu(
            weight_name(i, "post_attention_layernorm.weight"), cuda_alloc, DataType::kDataTypeFP32);

        auto wq = reader.get_tensor_gpu(
            weight_name(i, "self_attn.q_proj.weight"), cuda_alloc, DataType::kDataTypeFP16);
        auto wk = reader.get_tensor_gpu(
            weight_name(i, "self_attn.k_proj.weight"), cuda_alloc, DataType::kDataTypeFP16);
        auto wv = reader.get_tensor_gpu(
            weight_name(i, "self_attn.v_proj.weight"), cuda_alloc, DataType::kDataTypeFP16);
        auto wo = reader.get_tensor_gpu(
            weight_name(i, "self_attn.o_proj.weight"), cuda_alloc, DataType::kDataTypeFP16);

        auto w_gate = reader.get_tensor_gpu(
            weight_name(i, "mlp.gate_proj.weight"), cuda_alloc, DataType::kDataTypeFP16);
        auto w_up = reader.get_tensor_gpu(
            weight_name(i, "mlp.up_proj.weight"), cuda_alloc, DataType::kDataTypeFP16);
        auto w_down = reader.get_tensor_gpu(
            weight_name(i, "mlp.down_proj.weight"), cuda_alloc, DataType::kDataTypeFP16);

        // Qwen2 has Q/K/V bias
        auto bq = reader.get_tensor_gpu(
            weight_name(i, "self_attn.q_proj.bias"), cuda_alloc, DataType::kDataTypeFP16);
        auto bk = reader.get_tensor_gpu(
            weight_name(i, "self_attn.k_proj.bias"), cuda_alloc, DataType::kDataTypeFP16);
        auto bv = reader.get_tensor_gpu(
            weight_name(i, "self_attn.v_proj.bias"), cuda_alloc, DataType::kDataTypeFP16);

        blocks.push_back(std::make_shared<TransformerLayer>(
            cuda_alloc, c,
            attn_norm_w, wq, wk, wv, wo,
            ffn_norm_w, w_gate, w_up, w_down,
            true, bq, bk, bv));

        if ((i + 1) % 7 == 0)
            std::cout << "    Loaded " << i + 1 << "/" << c.layer << " layers\n";
    }

    auto final_norm_w = reader.get_tensor_gpu("model.norm.weight", cuda_alloc, DataType::kDataTypeFP32);

    std::shared_ptr<Tensor> lm_head_w;
    if (reader.has_tensor("lm_head.weight")) {
        lm_head_w = reader.get_tensor_gpu("lm_head.weight", cuda_alloc, DataType::kDataTypeFP16);
    } else {
        std::cout << ">>> [Loader] lm_head not found, reusing embed_tokens.weight\n";
        lm_head_w = embed_w;
    }

    auto embedding_layer = std::make_shared<EmbeddingLayer>(embed_w);
    auto final_norm = std::make_shared<RMSNormLayer>(final_norm_w);
    auto lm_head = std::make_shared<LinearLayer>(lm_head_w);

    auto model = std::make_shared<Qwen2Model>(
        cuda_alloc, embedding_layer, blocks, final_norm, lm_head, out_config);

    std::cout << ">>> [Loader] Qwen2 Model loaded!\n";
    return model;
}

} // namespace hxinfer
