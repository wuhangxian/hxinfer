
#include "loader/llama_tokenizer.h"
#include "stdexcept"
#include "iostream"
namespace hxinfer{
    LlamaTokenizer::LlamaTokenizer(const std::string &path, int vocab_size) {
        FILE *file=fopen(path.c_str(),"rb");
        if(!file){
            throw std::runtime_error("找不到tokenizer.bin文件");
        }
        int max_token_length;
        if (fread(&max_token_length, sizeof(int), 1, file) != 1) {
            throw std::runtime_error("读取 tokenizer 失败");
        }
        //读取分数和字符串长度
        //在 BPE（字节对编码） 算法中，每个 token 都有一个分数值（通常代表词频或合并优先级）。
        //推理引擎在编码（Encode）时，会根据这个分数决定如何将长句子切分成一个个 token。
        //在二进制文件中，字符串不是以 \0 结尾的，
        //而是采用 "Length-Prefixed"（长度前缀） 格式。程序必须先知道长度，才知道后面要读多少字节。
        for (int i = 0; i < vocab_size; i++) {
            float score;
            if (fread(&score, sizeof(float), 1, file) != 1) break;
            int len;
            if (fread(&len, sizeof(int), 1, file) != 1) break;

            std::string word(len, ' ');
            if (fread(&word[0], 1, len, file) != len) break;

            vocab_.push_back(word);
        }
        fclose(file);
        std::cout << "✅ LlamaTokenizer 字典加载成功！词表大小: " << vocab_.size() << std::endl;
    }

    std::string LlamaTokenizer::decode(int token_id) {
        if(token_id>=0&&token_id<vocab_.size()){
            std::string text=vocab_[token_id];
            // LLaMA 的特殊字符处理：把代表空格的特殊下划线替换回真正的空格
            if (text.length() >= 3 && text.substr(0, 3) == " ") {
                text = " " + text.substr(3);
            }
            return text;
        }
        return "";
    }
}