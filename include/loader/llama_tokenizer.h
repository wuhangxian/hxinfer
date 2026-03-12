#ifndef HXINFER_LLAMA_TOKENIZER_H
#define HXINFER_LLAMA_TOKENIZER_H
#include "vector"
#include "string"
namespace hxinfer{
    class LlamaTokenizer{
    private:
        std::vector<std::string> vocab_;
    public:
        LlamaTokenizer(const std::string& path,int vocab_size);

        std::string decode(int token_id);
    };
}

#endif //HXINFER_LLAMA_TOKENIZER_H
