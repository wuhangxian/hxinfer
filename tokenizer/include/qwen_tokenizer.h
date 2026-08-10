#ifndef HXINFER_QWEN_TOKENIZER_H
#define HXINFER_QWEN_TOKENIZER_H
#include <string>
#include <vector>
#include <unordered_map>
#include <nlohmann/json.hpp>

namespace hxinfer {

class QwenTokenizer {
private:
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<std::string, int> merge_rank_;
    std::unordered_map<int, std::string> id_to_token_;
    int bos_id_ = 151643;
    int eos_id_ = 151645;

    int byte_to_unicode_table_[256];
    int unicode_to_byte_table_[512];

    void init_byte_mapping();

public:
    explicit QwenTokenizer(const std::string& tokenizer_json_path);
    std::vector<int> encode(const std::string& text) const;
    std::string decode(int token_id) const;
    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }
};

} // namespace hxinfer
#endif
