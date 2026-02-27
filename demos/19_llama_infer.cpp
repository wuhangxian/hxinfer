#include "cstring"
#include "cstdlib"
#include "iostream"
#include "memory"
#include "vector"
#include "numeric"
#include "cmath"
#include <chrono>

struct Config {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
};

class Allocator {
public:
    virtual void* allocate(size_t byte_size) = 0;
    virtual void release(void* ptr) = 0;
    virtual ~Allocator() {}
};

class CPUAllocator : public Allocator {
public:
    void* allocate(size_t byte_size) override {
        void* ptr = nullptr;
        int ret = posix_memalign(&ptr, 64, byte_size);
        if (ret != 0 || ptr == nullptr) {
            std::cerr << "[Fatal] 内存大小不够, 无法申请到内存\n";
            exit(EXIT_FAILURE);
        }
        memset(ptr, 0, byte_size);
        return ptr;
    }
    void release(void* ptr) override {
        if (ptr != nullptr) free(ptr);
    }
};

class Buffer {
private:
    std::shared_ptr<Allocator> allocator_;
    size_t byte_size_;
    void* data_;
public:
    Buffer(size_t byte_size, std::shared_ptr<Allocator> allocator) : byte_size_(byte_size), allocator_(allocator) {
        data_ = allocator_->allocate(byte_size_);
    }
    ~Buffer() {
        if (data_ != nullptr) {
            allocator_->release(data_);
            data_ = nullptr;
        }
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    size_t buffer_byte_size() const { return byte_size_; }
    const void* buffer_data_ptr() const { return data_; }
    void* buffer_data_ptr() { return data_; }
};

class Tensor {
private:
    std::vector<int> shapes_;
    std::shared_ptr<Buffer> buffer_;
    size_t total_elements_;
public:
    Tensor(std::vector<int> shapes, std::shared_ptr<Allocator> allocator) : shapes_(shapes) {
        total_elements_ = std::accumulate(shapes_.begin(), shapes_.end(), size_t{1}, std::multiplies<>());
        size_t byte_size = total_elements_ * sizeof(float);
        buffer_ = std::make_shared<Buffer>(byte_size, allocator);
    }
    ~Tensor() {}
    size_t tensor_total_elements() const { return total_elements_; }
    void* tensor_data_ptr() { return buffer_->buffer_data_ptr(); }
};

struct LlamaWeights {
    std::shared_ptr<Tensor> token_embedding_table;
    std::shared_ptr<Tensor> rms_att_weight;
    std::shared_ptr<Tensor> wq, wk, wv, wo;
    std::shared_ptr<Tensor> rms_ffn_weight;
    std::shared_ptr<Tensor> w1, w2, w3;
    std::shared_ptr<Tensor> rms_final_weight;
    std::shared_ptr<Tensor> wcls;
};

class MathOps {
public:
    static void add_tensors(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, std::shared_ptr<Tensor> out) {
        float* a_ptr = (float*)a->tensor_data_ptr();
        float* b_ptr = (float*)b->tensor_data_ptr();
        float* out_ptr = (float*)out->tensor_data_ptr();
        for (size_t i = 0; i < a->tensor_total_elements(); i++) out_ptr[i] = a_ptr[i] + b_ptr[i];
    }
    static void mul_tensors(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, std::shared_ptr<Tensor> out) {
        float* a_ptr = (float*)a->tensor_data_ptr();
        float* b_ptr = (float*)b->tensor_data_ptr();
        float* out_ptr = (float*)out->tensor_data_ptr();
        for (size_t i = 0; i < a->tensor_total_elements(); i++) out_ptr[i] = a_ptr[i] * b_ptr[i];
    }
    static void silu_tensor(std::shared_ptr<Tensor> in, std::shared_ptr<Tensor> out) {
        float* in_ptr = (float*)in->tensor_data_ptr();
        float* out_ptr = (float*)out->tensor_data_ptr();
        for (size_t i = 0; i < in->tensor_total_elements(); i++) {
            out_ptr[i] = in_ptr[i] / (1.0f + std::exp(-in_ptr[i]));
        }
    }
};

class Layer {
public:
    std::string layer_name_;
    Layer(std::string name) : layer_name_(name) {}
    virtual void forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output) = 0;
    virtual ~Layer() {}
};

class RMSNormLayer : public Layer {
private:
    std::shared_ptr<Tensor> weights_;
    int dim_;
    float eps_ = 1e-5f;
public:
    RMSNormLayer(int dim, std::shared_ptr<Allocator> alloc) : Layer("RMSNorm"), dim_(dim) {
        weights_ = std::make_shared<Tensor>(std::vector<int>{dim}, alloc);
    }
    void copy_weights(std::shared_ptr<Tensor> all_layer_weights, int layer_idx) {
        float* dest = (float*)weights_->tensor_data_ptr();
        float* src = (float*)all_layer_weights->tensor_data_ptr() + layer_idx * dim_;
        memcpy(dest, src, dim_ * sizeof(float));
    }
    void forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output) override {
        int seq_len = input->tensor_total_elements() / dim_;
        float* in_ptr = (float*)input->tensor_data_ptr();
        float* out_ptr = (float*)output->tensor_data_ptr();
        float* w_ptr = (float*)weights_->tensor_data_ptr();

        for (int i = 0; i < seq_len; i++) {
            float* curr_in = in_ptr + i * dim_;
            float* curr_out = out_ptr + i * dim_;
            float sum = 0.0f;
            for (int j = 0; j < dim_; j++) sum += curr_in[j] * curr_in[j];
            float rms = 1.0f / std::sqrt(sum / dim_ + eps_);
            for (int j = 0; j < dim_; j++) curr_out[j] = curr_in[j] * rms * w_ptr[j];
        }
    }
};

class LinearLayer : public Layer {
private:
    std::shared_ptr<Tensor> weights_;
    int in_features_;
    int out_features_;
public:
    LinearLayer(int in_features, int out_features, std::shared_ptr<Allocator> alloc)
            : Layer("Linear"), in_features_(in_features), out_features_(out_features) {
        weights_ = std::make_shared<Tensor>(std::vector<int>{out_features_, in_features_}, alloc);
    }
    void copy_weights(std::shared_ptr<Tensor> all_layer_weights, int layer_idx) {
        size_t layer_elements = in_features_ * out_features_;
        float* dest = (float*)weights_->tensor_data_ptr();
        float* src = (float*)all_layer_weights->tensor_data_ptr() + layer_idx * layer_elements;
        memcpy(dest, src, layer_elements * sizeof(float));
    }
    void forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output) override {
        int seq_len = input->tensor_total_elements() / in_features_;
        float* in_ptr = (float*)input->tensor_data_ptr();
        float* w_ptr = (float*)weights_->tensor_data_ptr();
        float* out_ptr = (float*)output->tensor_data_ptr();

        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < out_features_; j++) {
                float sum = 0.0f;
                for (int k = 0; k < in_features_; k++) {
                    sum += in_ptr[i * in_features_ + k] * w_ptr[j * in_features_ + k];
                }
                out_ptr[i * out_features_ + j] = sum;
            }
        }
    }
};

// ==========================================
// 🚀 极其暴力的全量 Naive Attention
// 每次都把历史全部算一遍，没有任何缓存！
// ==========================================
class LlamaAttention : public Layer {
private:
    int dim_;
    int n_heads_;
    int head_size_;
    std::shared_ptr<Allocator> alloc_;
    std::shared_ptr<LinearLayer> wq_, wk_, wv_, wo_;

public:
    LlamaAttention(Config* config, LlamaWeights* weights, int layer_idx, std::shared_ptr<Allocator> alloc)
            : Layer("LlamaAttention"), dim_(config->dim), n_heads_(config->n_heads), alloc_(alloc) {
        head_size_ = dim_ / n_heads_;
        wq_ = std::make_shared<LinearLayer>(dim_, dim_, alloc); wq_->copy_weights(weights->wq, layer_idx);
        wk_ = std::make_shared<LinearLayer>(dim_, dim_, alloc); wk_->copy_weights(weights->wk, layer_idx);
        wv_ = std::make_shared<LinearLayer>(dim_, dim_, alloc); wv_->copy_weights(weights->wv, layer_idx);
        wo_ = std::make_shared<LinearLayer>(dim_, dim_, alloc); wo_->copy_weights(weights->wo, layer_idx);
    }

    void forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output) override {
        int seq_len = input->tensor_total_elements() / dim_;

        // 🌟 Naive 核心：每次都硬碰硬地动态申请跟 seq_len 一样长的新内存！
        auto q_buf = std::make_shared<Tensor>(std::vector<int>{seq_len, dim_}, alloc_);
        auto k_buf = std::make_shared<Tensor>(std::vector<int>{seq_len, dim_}, alloc_);
        auto v_buf = std::make_shared<Tensor>(std::vector<int>{seq_len, dim_}, alloc_);
        auto att_out_buf = std::make_shared<Tensor>(std::vector<int>{seq_len, dim_}, alloc_);

        wq_->forward(input, q_buf);
        wk_->forward(input, k_buf);
        wv_->forward(input, v_buf);

        float* q_ptr = (float*)q_buf->tensor_data_ptr();
        float* k_ptr = (float*)k_buf->tensor_data_ptr();
        float* v_ptr = (float*)v_buf->tensor_data_ptr();
        float* out_ptr = (float*)att_out_buf->tensor_data_ptr();

        // 🌟 暴力的全量 RoPE 注入
        for (int p = 0; p < seq_len; p++) {
            for (int i = 0; i < dim_; i += 2) {
                int head_dim = i % head_size_;
                float freq = 1.0f / std::pow(10000.0f, (float)head_dim / (float)head_size_);
                float val = p * freq;
                float fcr = std::cos(val), fci = std::sin(val);

                float q0 = q_ptr[p * dim_ + i], q1 = q_ptr[p * dim_ + i + 1];
                q_ptr[p * dim_ + i]     = q0 * fcr - q1 * fci;
                q_ptr[p * dim_ + i + 1] = q0 * fci + q1 * fcr;

                float k0 = k_ptr[p * dim_ + i], k1 = k_ptr[p * dim_ + i + 1];
                k_ptr[p * dim_ + i]     = k0 * fcr - k1 * fci;
                k_ptr[p * dim_ + i + 1] = k0 * fci + k1 * fcr;
            }
        }

        // 🌟 暴力的全量 O(N^2) 矩阵乘法
        float scale = 1.0f / std::sqrt(head_size_);
        for (int h = 0; h < n_heads_; h++) {
            for (int i = 0; i < seq_len; i++) {
                std::vector<float> scores(seq_len, 0.0f);
                float max_score = -1e9f;

                // 因果掩码：只看前面的历史，不看未来的词
                for (int j = 0; j <= i; j++) {
                    float sum = 0.0f;
                    for (int d = 0; d < head_size_; d++) {
                        sum += q_ptr[i * dim_ + h * head_size_ + d] * k_ptr[j * dim_ + h * head_size_ + d];
                    }
                    scores[j] = sum * scale;
                    if (scores[j] > max_score) max_score = scores[j];
                }

                float exp_sum = 0.0f;
                for (int j = 0; j <= i; j++) {
                    scores[j] = std::exp(scores[j] - max_score);
                    exp_sum += scores[j];
                }

                for (int d = 0; d < head_size_; d++) {
                    float sum = 0.0f;
                    for (int j = 0; j <= i; j++) {
                        sum += (scores[j] / exp_sum) * v_ptr[j * dim_ + h * head_size_ + d];
                    }
                    out_ptr[i * dim_ + h * head_size_ + d] = sum;
                }
            }
        }

        wo_->forward(att_out_buf, output);
    }
};

class TransformerBlock : public Layer {
private:
    int layer_idx_, dim_, hidden_dim_;
    std::shared_ptr<Allocator> alloc_;
    std::shared_ptr<LlamaAttention> attention_;
    std::shared_ptr<RMSNormLayer> rms_att_, rms_ffn_;
    std::shared_ptr<LinearLayer> w1_, w2_, w3_;

public:
    TransformerBlock(int layer_idx, Config* config, LlamaWeights* weights, std::shared_ptr<Allocator> alloc)
            : Layer("TransformerBlock"), layer_idx_(layer_idx), dim_(config->dim), hidden_dim_(config->hidden_dim), alloc_(alloc) {

        attention_ = std::make_shared<LlamaAttention>(config, weights, layer_idx_, alloc_);
        rms_att_ = std::make_shared<RMSNormLayer>(dim_, alloc_); rms_att_->copy_weights(weights->rms_att_weight, layer_idx_);
        rms_ffn_ = std::make_shared<RMSNormLayer>(dim_, alloc_); rms_ffn_->copy_weights(weights->rms_ffn_weight, layer_idx_);
        w1_ = std::make_shared<LinearLayer>(dim_, hidden_dim_, alloc_); w1_->copy_weights(weights->w1, layer_idx_);
        w2_ = std::make_shared<LinearLayer>(hidden_dim_, dim_, alloc_); w2_->copy_weights(weights->w2, layer_idx_);
        w3_ = std::make_shared<LinearLayer>(dim_, hidden_dim_, alloc_); w3_->copy_weights(weights->w3, layer_idx_);
    }

    void forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output) override {
        int seq_len = input->tensor_total_elements() / dim_;

        // 🌟 Naive 核心：每经过一层，都要疯狂申请一次物理内存
        auto norm_att_out = std::make_shared<Tensor>(std::vector<int>{seq_len, dim_}, alloc_);
        auto att_out = std::make_shared<Tensor>(std::vector<int>{seq_len, dim_}, alloc_);
        auto h_tensor = std::make_shared<Tensor>(std::vector<int>{seq_len, dim_}, alloc_);
        auto norm_ffn_out = std::make_shared<Tensor>(std::vector<int>{seq_len, dim_}, alloc_);
        auto w1_out = std::make_shared<Tensor>(std::vector<int>{seq_len, hidden_dim_}, alloc_);
        auto w3_out = std::make_shared<Tensor>(std::vector<int>{seq_len, hidden_dim_}, alloc_);
        auto ffn_out = std::make_shared<Tensor>(std::vector<int>{seq_len, dim_}, alloc_);

        rms_att_->forward(input, norm_att_out);
        attention_->forward(norm_att_out, att_out);
        MathOps::add_tensors(input, att_out, h_tensor);

        rms_ffn_->forward(h_tensor, norm_ffn_out);
        w1_->forward(norm_ffn_out, w1_out);
        MathOps::silu_tensor(w1_out, w1_out);
        w3_->forward(norm_ffn_out, w3_out);
        MathOps::mul_tensors(w1_out, w3_out, w1_out);
        w2_->forward(w1_out, ffn_out);

        MathOps::add_tensors(h_tensor, ffn_out, output);
    }
};

class LlamaModel {
private:
    Config config_;
    LlamaWeights* weights_;
    std::shared_ptr<Allocator> alloc_;
    std::vector<std::shared_ptr<TransformerBlock>> blocks_;
    std::shared_ptr<RMSNormLayer> rms_final_;

public:
    LlamaModel(Config config, LlamaWeights* weights, std::shared_ptr<Allocator> alloc)
            : config_(config), weights_(weights), alloc_(alloc) {
        for (int i = 0; i < config_.n_layers; i++) {
            blocks_.push_back(std::make_shared<TransformerBlock>(i, &config_, weights_, alloc_));
        }
        rms_final_ = std::make_shared<RMSNormLayer>(config_.dim, alloc_);
        rms_final_->copy_weights(weights_->rms_final_weight, 0);
    }

    // 🌟 暴改接口：每次传入整个历史词表！
    int forward(const std::vector<int>& history_tokens) {
        int seq_len = history_tokens.size();

        // 动态申请涵盖整个历史的长 Tensor
        auto x = std::make_shared<Tensor>(std::vector<int>{seq_len, config_.dim}, alloc_);
        float* x_ptr = (float*)x->tensor_data_ptr();
        float* emb_table = (float*)weights_->token_embedding_table->tensor_data_ptr();

        // 把历史词表全部从 Embedding 字典里查出来，拼成长序列
        for (int i = 0; i < seq_len; i++) {
            memcpy(x_ptr + i * config_.dim, emb_table + history_tokens[i] * config_.dim, config_.dim * sizeof(float));
        }

        // 穿越 6 层机甲 (全量序列穿越)
        for (int i = 0; i < config_.n_layers; i++) {
            blocks_[i]->forward(x, x);
        }
        rms_final_->forward(x, x);

        // 🌟 核心：只抽取最后一个词的特征用于分类预测！
        float* last_token_features = x_ptr + (seq_len - 1) * config_.dim;
        float* wcls_ptr = (float*)weights_->wcls->tensor_data_ptr();

        int best_token = 0;
        float max_score = -1e9f;

        for (int v = 0; v < config_.vocab_size; v++) {
            float score = 0.0f;
            for (int d = 0; d < config_.dim; d++) {
                score += last_token_features[d] * wcls_ptr[v * config_.dim + d];
            }
            if (score > max_score) {
                max_score = score;
                best_token = v;
            }
        }
        return best_token;
    }
};

class Tokenizer {
private:
    std::vector<std::string> vocab_;
public:
    Tokenizer(const char* tokenizer_path, int vocab_size) {
        FILE* file = fopen(tokenizer_path, "rb");
        if (!file) {
            std::cerr << "[Fatal] 找不到词表文件\n";
            exit(EXIT_FAILURE);
        }
        int max_token_length;
        fread(&max_token_length, sizeof(int), 1, file);
        vocab_.resize(vocab_size);
        char word_buffer[256];
        for (int i = 0; i < vocab_size; i++) {
            float score; fread(&score, sizeof(float), 1, file);
            int len; fread(&len, sizeof(int), 1, file);
            fread(word_buffer, 1, len, file);
            word_buffer[len] = '\0';
            vocab_[i] = std::string(word_buffer);
        }
        fclose(file);
    }
    std::string decode(int token_id) {
        if (token_id >= 0 && token_id < vocab_.size()) return vocab_[token_id];
        return "";
    }
};

void load_weights(const char* checkpoint_path, Config* config, LlamaWeights* weights, std::shared_ptr<Allocator> alloc) {
    FILE* file = fopen(checkpoint_path, "rb");
    if (!file) exit(EXIT_FAILURE);
    fread(config, sizeof(Config), 1, file);
    int dim = config->dim, hidden_dim = config->hidden_dim, layers = config->n_layers, vocab_size = config->vocab_size;
    weights->token_embedding_table = std::make_shared<Tensor>(std::vector<int>{vocab_size, dim}, alloc);
    weights->rms_att_weight = std::make_shared<Tensor>(std::vector<int>{layers, dim}, alloc);
    weights->wq = std::make_shared<Tensor>(std::vector<int>{layers, dim, dim}, alloc);
    weights->wk = std::make_shared<Tensor>(std::vector<int>{layers, dim, dim}, alloc);
    weights->wv = std::make_shared<Tensor>(std::vector<int>{layers, dim, dim}, alloc);
    weights->wo = std::make_shared<Tensor>(std::vector<int>{layers, dim, dim}, alloc);
    weights->rms_ffn_weight = std::make_shared<Tensor>(std::vector<int>{layers, dim}, alloc);
    weights->w1 = std::make_shared<Tensor>(std::vector<int>{layers, hidden_dim, dim}, alloc);
    weights->w2 = std::make_shared<Tensor>(std::vector<int>{layers, dim, hidden_dim}, alloc);
    weights->w3 = std::make_shared<Tensor>(std::vector<int>{layers, hidden_dim, dim}, alloc);
    weights->rms_final_weight = std::make_shared<Tensor>(std::vector<int>{dim}, alloc);
    weights->wcls = std::make_shared<Tensor>(std::vector<int>{vocab_size, dim}, alloc);

    fread(weights->token_embedding_table->tensor_data_ptr(), sizeof(float), vocab_size * dim, file);
    fread(weights->rms_att_weight->tensor_data_ptr(), sizeof(float), layers * dim, file);
    fread(weights->wq->tensor_data_ptr(), sizeof(float), layers * dim * dim, file);
    fread(weights->wk->tensor_data_ptr(), sizeof(float), layers * dim * dim, file);
    fread(weights->wv->tensor_data_ptr(), sizeof(float), layers * dim * dim, file);
    fread(weights->wo->tensor_data_ptr(), sizeof(float), layers * dim * dim, file);
    fread(weights->rms_ffn_weight->tensor_data_ptr(), sizeof(float), layers * dim, file);
    fread(weights->w1->tensor_data_ptr(), sizeof(float), layers * hidden_dim * dim, file);
    fread(weights->w2->tensor_data_ptr(), sizeof(float), layers * dim * hidden_dim, file);
    fread(weights->w3->tensor_data_ptr(), sizeof(float), layers * hidden_dim * dim, file);
    fread(weights->rms_final_weight->tensor_data_ptr(), sizeof(float), dim, file);

    int head_size = dim / config->n_heads;
    fseek(file, config->seq_len * (head_size / 2) * sizeof(float) * 2, SEEK_CUR);
    if (fread(weights->wcls->tensor_data_ptr(), sizeof(float), vocab_size * dim, file) == 0) {
        weights->wcls = weights->token_embedding_table;
    }
    fclose(file);
}

int main() {
    std::cout << "--- [纯粹暴力] 无 KV Cache 的全量上下文序列生成 ---" << '\n';

    const char* checkpoint_path = "/home/whx/hxinfer/models/stories15M.bin";
    const char* tokenizer_path = "/home/whx/hxinfer/models/tokenizer.bin";

    Config config;
    std::shared_ptr<Allocator> alloc = std::make_shared<CPUAllocator>();
    LlamaWeights weights;

    load_weights(checkpoint_path, &config, &weights, alloc);
    Tokenizer tokenizer(tokenizer_path, config.vocab_size);
    LlamaModel model(config, &weights, alloc);

    std::cout << "\n============================================\n";
    std::cout << "童话故事开始生成 (附带实时 Token/s 监测)：\n\n";

    std::vector<int> history;
    history.push_back(1); // 塞入 <s> 起始符

    int target_steps = 200;

    // 🌟 记录整场推理的总开始时间
    auto start_all = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < target_steps; step++) {

        // 🌟 记录单个 Token 推理的开始时间
        auto start_step = std::chrono::high_resolution_clock::now();

        // 极其耗时的全量计算
        int next_token = model.forward(history);

        // 🌟 记录单个 Token 推理的结束时间，并计算瞬间速度
        auto end_step = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> step_duration = end_step - start_step;
        double step_tok_per_sec = 1.0 / step_duration.count();

        std::string word = tokenizer.decode(next_token);

        // 打印生成的词，并在后面带上它消耗的瞬间速度
        // 加上 \033 颜色代码让速度显示为灰色，不影响阅读英文
        std::cout << word << " \033[90m[" << step_tok_per_sec << " tok/s]\033[0m";
        fflush(stdout);

        history.push_back(next_token);
    }

    // 🌟 记录整场推理的总结束时间，计算宏观平均速度
    auto end_all = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = end_all - start_all;
    double avg_tok_per_sec = target_steps / total_duration.count();

    std::cout << "\n\n============================================\n";
    std::cout << ">>> 推理性能测试报告：\n";
    std::cout << "  - 生成总 Token 数: " << target_steps << "\n";
    std::cout << "  - 总耗时: " << total_duration.count() << " 秒\n";
    std::cout << "  - 宏观平均速度: " << avg_tok_per_sec << " Token/s\n";
    std::cout << "============================================\n";

    return 0;
}