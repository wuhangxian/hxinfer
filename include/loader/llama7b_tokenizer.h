#ifndef HXINFER_LLAMA7B_TOKENIZER_H
#define HXINFER_LLAMA7B_TOKENIZER_H
#include <string>
#include <vector>
#include <memory>
#include <sentencepiece_processor.h>

namespace hxinfer{
    class Llama7BTokenizer{
    private:
        sentencepiece::SentencePieceProcessor sp_;
    public:
        explicit Llama7BTokenizer(const std::string& model_path);
        std::vector<int> encode(const std::string& text) const;
        std::string decode(int token_id) const;
        int bos_id() const { return sp_.bos_id(); }
        int eos_id() const { return sp_.eos_id(); }
    };
}

#endif
