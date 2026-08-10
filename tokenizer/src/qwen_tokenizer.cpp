#include "qwen_tokenizer.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstring>

namespace hxinfer {

using json = nlohmann::json;

static const int* get_byte_to_unicode() {
    static int b2u[256];
    static bool init = false;
    if (init) return b2u;

    std::vector<int> bs;
    for (int b = (int)'!'; b <= (int)'~'; b++) bs.push_back(b);
    for (int b = 0xA1; b <= 0x1AC; b++) bs.push_back(b);
    for (int b = 0x2000; b <= 0x206F; b++) bs.push_back(b);
    for (int b = 0x2E80; b <= 0x2FFF; b++) bs.push_back(b);

    std::vector<int> cs(bs.begin(), bs.end());
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    for (int i = 0; i < 256; i++) b2u[i] = i;
    for (int i = 0; i < (int)bs.size(); i++) { if (bs[i] < 256) b2u[bs[i]] = cs[i]; }

    init = true;
    return b2u;
}

static std::string unicode_to_utf8(int cp) {
    std::string out;
    if (cp <= 0x7F) { out += (char)cp; }
    else if (cp <= 0x7FF) {
        out += (char)(0xC0 | (cp >> 6));
        out += (char)(0x80 | (cp & 0x3F));
    } else if (cp <= 0xFFFF) {
        out += (char)(0xE0 | (cp >> 12));
        out += (char)(0x80 | ((cp >> 6) & 0x3F));
        out += (char)(0x80 | (cp & 0x3F));
    } else {
        out += (char)(0xF0 | (cp >> 18));
        out += (char)(0x80 | ((cp >> 12) & 0x3F));
        out += (char)(0x80 | ((cp >> 6) & 0x3F));
        out += (char)(0x80 | (cp & 0x3F));
    }
    return out;
}

static int utf8_next(const std::string& s, size_t& i) {
    if (i >= s.size()) return -1;
    unsigned char c = s[i];
    if (c <= 0x7F) { i++; return c; }
    if ((c & 0xE0) == 0xC0) {
        int cp = (c & 0x1F) << 6;
        if (i + 1 < s.size()) cp |= (unsigned char)s[i+1] & 0x3F;
        i += 2; return cp;
    }
    if ((c & 0xF0) == 0xE0) {
        int cp = (c & 0x0F) << 12;
        if (i + 1 < s.size()) cp |= ((unsigned char)s[i+1] & 0x3F) << 6;
        if (i + 2 < s.size()) cp |= (unsigned char)s[i+2] & 0x3F;
        i += 3; return cp;
    }
    if ((c & 0xF8) == 0xF0) {
        int cp = (c & 0x07) << 18;
        if (i + 1 < s.size()) cp |= ((unsigned char)s[i+1] & 0x3F) << 12;
        if (i + 2 < s.size()) cp |= ((unsigned char)s[i+2] & 0x3F) << 6;
        if (i + 3 < s.size()) cp |= (unsigned char)s[i+3] & 0x3F;
        i += 4; return cp;
    }
    i++;
    return c;
}

static std::string byte_encode(const std::string& text) {
    const int* b2u = get_byte_to_unicode();
    std::string result;
    for (size_t i = 0; i < text.size(); i++) {
        unsigned char c = text[i];
        result += unicode_to_utf8(b2u[c]);
    }
    return result;
}

QwenTokenizer::QwenTokenizer(const std::string& tokenizer_json_path) {
    std::ifstream f(tokenizer_json_path);
    if (!f) throw std::runtime_error("Cannot open tokenizer.json: " + tokenizer_json_path);
    json tok = json::parse(f);

    json model = tok["model"];
    json vocab_json = model["vocab"];
    json merges_json = model["merges"];

    for (auto& [key, val] : vocab_json.items()) {
        vocab_[key] = val.get<int>();
        id_to_token_[val.get<int>()] = key;
    }

    for (int i = 0; i < (int)merges_json.size(); i++) {
        merge_rank_[merges_json[i].get<std::string>()] = i;
    }

    if (tok.contains("added_tokens")) {
        for (auto& at : tok["added_tokens"]) {
            std::string content = at["content"].get<std::string>();
            int id = at["id"].get<int>();
            vocab_[content] = id;
            id_to_token_[id] = content;
        }
    }

    auto it = vocab_.find("");
    if (it != vocab_.end()) eos_id_ = it->second;
}

// Pre-tokenizer: matches Qwen2/GPT2 regex pattern
// Simplified: split into words/punctuation, preserving leading spaces with next word
static std::vector<std::string> pre_tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = text[i];

        if (c == ' ') {
            // Find the next word after spaces
            size_t space_start = i;
            while (i < text.size() && text[i] == ' ') i++;
            if (i < text.size() && ((text[i] >= 'a' && text[i] <= 'z') || (text[i] >= 'A' && text[i] <= 'Z'))) {
                // Attach one space to the following word
                size_t word_start = i;
                while (i < text.size() && ((text[i] >= 'a' && text[i] <= 'z') || (text[i] >= 'A' && text[i] <= 'Z'))) i++;
                std::string tok = " " + text.substr(word_start, i - word_start);
                tokens.push_back(tok);
            } else if (i < text.size() && text[i] >= '0' && text[i] <= '9') {
                size_t num_start = i;
                while (i < text.size() && text[i] >= '0' && text[i] <= '9') i++;
                std::string tok = " " + text.substr(num_start, i - num_start);
                tokens.push_back(tok);
            } else if (i < text.size()) {
                // Attach space to punctuation
                size_t punct_start = i;
                while (i < text.size()) {
                    unsigned char ch = text[i];
                    if (ch == ' ' || (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9')) break;
                    i++;
                }
                std::string tok = " " + text.substr(punct_start, i - punct_start);
                tokens.push_back(tok);
            } else {
                // Trailing spaces
                tokens.push_back(text.substr(space_start, i - space_start));
            }
        } else if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
            size_t start = i;
            while (i < text.size() && ((text[i] >= 'a' && text[i] <= 'z') || (text[i] >= 'A' && text[i] <= 'Z'))) i++;
            tokens.push_back(text.substr(start, i - start));
        } else if (c >= '0' && c <= '9') {
            size_t start = i;
            while (i < text.size() && text[i] >= '0' && text[i] <= '9') i++;
            tokens.push_back(text.substr(start, i - start));
        } else {
            size_t start = i;
            while (i < text.size()) {
                unsigned char ch = text[i];
                if (ch == ' ' || (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9')) break;
                i++;
            }
            tokens.push_back(text.substr(start, i - start));
        }
    }
    return tokens;
}

static std::vector<std::string> bpe_merge(
    const std::string& token,
    const std::unordered_map<std::string, int>& merge_rank)
{
    if (token.size() <= 1) return {token};

    std::vector<std::string> symbols;
    for (size_t i = 0; i < token.size(); ) {
        size_t j = i + 1;
        while (j < token.size() && (token[j] & 0xC0) == 0x80) j++;
        symbols.push_back(token.substr(i, j - i));
        i = j;
    }

    if (symbols.size() <= 1) return symbols;

    while (symbols.size() > 1) {
        int best_rank = 0x7FFFFFFF;
        int best_idx = -1;
        for (int j = 0; j < (int)symbols.size() - 1; j++) {
            std::string pair = symbols[j] + " " + symbols[j+1];
            auto it = merge_rank.find(pair);
            if (it != merge_rank.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = j;
            }
        }
        if (best_idx < 0) break;

        std::vector<std::string> new_symbols;
        for (int j = 0; j < (int)symbols.size(); j++) {
            if (j == best_idx) {
                new_symbols.push_back(symbols[j] + symbols[j+1]);
                j++;
            } else {
                new_symbols.push_back(symbols[j]);
            }
        }
        symbols = new_symbols;
    }
    return symbols;
}

std::vector<int> QwenTokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    std::vector<std::string> pre_tokens = pre_tokenize(text);

    for (const auto& pt : pre_tokens) {
        std::string encoded = byte_encode(pt);

        auto direct = vocab_.find(encoded);
        if (direct != vocab_.end()) {
            ids.push_back(direct->second);
            continue;
        }

        std::vector<std::string> sub_tokens = bpe_merge(encoded, merge_rank_);
        for (const auto& st : sub_tokens) {
            auto it = vocab_.find(st);
            if (it != vocab_.end()) ids.push_back(it->second);
        }
    }
    return ids;
}

std::string QwenTokenizer::decode(int token_id) const {
    auto it = id_to_token_.find(token_id);
    if (it == id_to_token_.end()) return "";

    std::string token = it->second;
    const int* b2u = get_byte_to_unicode();
    int u2b[512];
    for (int i = 0; i < 512; i++) u2b[i] = -1;
    for (int i = 0; i < 256; i++) u2b[b2u[i]] = i;

    std::string result;
    for (size_t i = 0; i < token.size(); ) {
        int cp = utf8_next(token, i);
        int byte_val = (cp < 512) ? u2b[cp] : -1;
        if (byte_val >= 0) {
            result += (char)(unsigned char)byte_val;
        } else {
            result += unicode_to_utf8(cp);
        }
    }
    return result;
}

} // namespace hxinfer
