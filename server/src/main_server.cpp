#include <iostream>
#include <string>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include "httplib.h"
#include "inference_engine.h"

using namespace hxinfer;
using json = nlohmann::json;

int main(int argc, char** argv) {
    int port = 8000;
    std::string model_dir;
    std::string tokenizer_path;
    ModelType type = ModelType::LLaMA;

    // Parse args: --model-dir /data/models/Llama-2-7b-hf --port 8000 --type llama
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model-dir" && i+1 < argc) model_dir = argv[++i];
        else if (arg == "--tokenizer" && i+1 < argc) tokenizer_path = argv[++i];
        else if (arg == "--port" && i+1 < argc) port = std::atoi(argv[++i]);
        else if (arg == "--type" && i+1 < argc) {
            std::string t = argv[++i];
            if (t == "qwen") type = ModelType::Qwen2;
        } else if (arg == "--help") {
            std::cout << "Usage: hxinfer-server --model-dir <path> --tokenizer <path> [--port 8000] [--type llama|qwen]" << std::endl;
            return 0;
        }
    }

    // Fall back to env vars
    const char* env_data = std::getenv("HXINFER_DATA_DIR");
    if (model_dir.empty()) model_dir = env_data ? env_data : "/data/models/Llama-2-7b-hf";
    if (tokenizer_path.empty()) {
        if (type == ModelType::Qwen2)
            tokenizer_path = model_dir + "/tokenizer.json";
        else
            tokenizer_path = model_dir + "/tokenizer.model";
    }

    std::cout << "==================================================" << std::endl;
    std::cout << "  hxinfer HTTP Server" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Model: " << model_dir << std::endl;
    std::cout << "Tokenizer: " << tokenizer_path << std::endl;
    std::cout << "Port: " << port << std::endl;
    std::cout << std::endl;

    InferenceEngine engine;
    engine.load(model_dir, type, tokenizer_path,
                type == ModelType::Qwen2 ? "qwen" : "llama");

    httplib::Server svr;

    // POST /generate
    svr.Post("/generate", [&engine](const httplib::Request& req, httplib::Response& res) {
        json req_json;
        try {
            req_json = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error": "Invalid JSON"})", "application/json");
            return;
        }

        std::string prompt = req_json.value("text", "");
        if (prompt.empty()) {
            res.status = 400;
            res.set_content(R"({"error": "Missing 'text' field"})", "application/json");
            return;
        }

        json sp = req_json.value("sampling_params", json::object());
        int max_new_tokens = sp.value("max_new_tokens", 64);
        float temperature = sp.value("temperature", 0.0f);
        float top_p = sp.value("top_p", 1.0f);

        auto result = engine.generate(prompt, max_new_tokens, temperature, top_p);

        json resp;
        resp["text"] = prompt + result.text;
        resp["generated_text"] = result.text;
        resp["token_ids"] = result.token_ids;
        resp["prompt_tokens"] = result.prompt_tokens;
        resp["generated_tokens"] = result.generated_tokens;
        resp["prefill_time_ms"] = result.prefill_time_ms;
        resp["decode_time_ms"] = result.decode_time_ms;
        resp["total_time_ms"] = result.total_time_ms;
        resp["tokens_per_second"] = result.generated_tokens / (result.decode_time_ms / 1000.0);

        res.set_content(resp.dump(), "application/json");
    });

    // GET /v1/models (OpenAI compatible — needed by bench_serving)
    svr.Get("/v1/models", [&engine](const httplib::Request&, httplib::Response& res) {
        json resp;
        json model_obj;
        model_obj["id"] = "hxinfer";
        model_obj["object"] = "model";
        model_obj["owned_by"] = "hxinfer";
        resp["object"] = "list";
        resp["data"] = json::array({model_obj});
        res.set_content(resp.dump(), "application/json");
    });

    // GET /health
    svr.Get("/health", [&engine](const httplib::Request&, httplib::Response& res) {
        json resp;
        resp["status"] = engine.is_loaded() ? "ready" : "loading";
        res.set_content(resp.dump(), "application/json");
    });

    // GET /model_info
    svr.Get("/model_info", [&engine](const httplib::Request&, httplib::Response& res) {
        json resp;
        resp["loaded"] = engine.is_loaded();
        res.set_content(resp.dump(), "application/json");
    });

    std::cout << "Server listening on http://0.0.0.0:" << port << std::endl;
    svr.listen("0.0.0.0", port);
    return 0;
}
