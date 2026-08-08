#include <iostream>
#include <vector>
#include "qwen_tokenizer.h"

int main() {
    hxinfer::QwenTokenizer tok("/root/dorianwu/models/Qwen2.5-7B-Instruct/tokenizer.json");
    auto ids = tok.encode("Once upon a time");
    std::cout << "Encode: ";
    for (int id : ids) std::cout << id << " ";
    std::cout << std::endl;
    std::cout << "Decode: ";
    for (int id : ids) std::cout << "[" << tok.decode(id) << "]";
    std::cout << std::endl;
    return 0;
}
