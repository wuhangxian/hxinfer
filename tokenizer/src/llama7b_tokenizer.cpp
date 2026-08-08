#include "llama7b_tokenizer.h"
#include <stdexcept>

namespace hxinfer{
    Llama7BTokenizer::Llama7BTokenizer(const std::string& model_path){
        auto status = sp_.Load(model_path);
        if(!status.ok()){
            throw std::runtime_error("加载 tokenizer.model 失败: " + model_path);
        }
    }

    std::vector<int> Llama7BTokenizer::encode(const std::string& text) const{
        std::vector<int> ids;
        sp_.Encode(text, &ids);
        return ids;
    }

    std::string Llama7BTokenizer::decode(int token_id) const{
        std::string piece = sp_.IdToPiece(token_id);

        // 处理字节 token，如 <0x0A> → 实际字节 '\n'
        if(piece.size()==6 && piece[0]=='<' && piece[1]=='0' && piece[2]=='x' && piece[5]=='>'){
            std::string hex = piece.substr(1,4);  // "0x0A"
            int byte_val = std::stoi(hex, nullptr, 16);
            return std::string(1, (char)byte_val);
        }

        // SentencePiece 用 '▁'（U+2581）表示空格，替换回真正的空格
        std::string result;
        for(size_t i=0; i<piece.size();){
            // '▁' 在 UTF-8 中是 3 字节: 0xE2 0x96 0x81
            if(i+2 < piece.size() &&
               (unsigned char)piece[i]==0xE2 &&
               (unsigned char)piece[i+1]==0x96 &&
               (unsigned char)piece[i+2]==0x81){
                result += ' ';
                i += 3;
            } else {
                result += piece[i++];
            }
        }
        return result;
    }
}
