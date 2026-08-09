# hxinfer v1.0.0 — 从零构建的 C++/CUDA LLM 推理引擎

从零手写的高性能 LLM 推理引擎，基于 C++/CUDA，支持 LLaMA-2、Qwen2.5 等主流 Dense Transformer 模型。纯 C++/CUDA 实现，零第三方深度学习框架依赖。

## 特性

- **多模型支持** — LLaMA-2-7B、Qwen2.5-7B-Instruct，同一引擎运行不同架构家族
- **通用 safetensors 加载** — 直接读取 HuggingFace safetensors 格式，零预处理，支持多 shard
- **GQA 支持** — 正确处理 Grouped Query Attention（多 Q head 共享 KV head）
- **Attention Bias 支持** — 自动检测 Q/K/V bias（Qwen2.5 有，LLaMA-2 无），有则加载无则跳过
- **可配置 RoPE** — 支持 rope_theta 参数（LLaMA-2: 10000，Qwen2.5: 1000000）+ YaRN 缩放
- **双后端支持** — CPU（纯 C++）和 CUDA（手写 Kernel + cuBLAS），运行时自动分发
- **FP16 推理** — 支持 FP16/BF16 → FP16 自动转换，与 PyTorch 行为一致
- **零框架依赖** — 不依赖 PyTorch/TensorFlow，仅需 C++17 编译器 + CUDA Toolkit
- **算子融合** — 融合 QKV 投影、融合 Gate+Up 投影、融合 SwiGLU
- **mmap 零拷贝加载** — 模型权重通过内存映射直接加载
- **KV Cache** — 自回归解码时缓存历史 K/V
- **前缀缓存（Prefix Cache）** — KV Cache 快照与恢复，相同前缀 token 跳过重复计算
- **Temperature + Top-p 采样** — 支持贪心解码和温度采样
- **双 Tokenizer** — SentencePiece（LLaMA）+ Byte-level BPE（Qwen）
- **RAII 内存管理** — DeviceAllocator → Buffer → Tensor 三层架构，自动内存生命周期管理

## 支持的模型

| 模型 | 架构 | KV Heads | Attention Bias | RoPE Theta | Tokenizer | 状态 |
|------|------|----------|---------------|------------|-----------|------|
| LLaMA-2-7B | LlamaForCausalLM | 32 (MHA) | 无 | 10000 (YaRN) | SentencePiece | ✅ 已验证 |
| Qwen2.5-7B-Instruct | Qwen2ForCausalLM | 4 (GQA) | 有 | 1000000 | BPE (tokenizer.json) | ✅ 已验证 |
| LLaMA-3-8B | LlamaForCausalLM | 8 (GQA) | 无 | 500000 | BPE | ✅ 兼容（架构相同） |
| Mistral-7B | MistralForCausalLM | 8 (GQA) | 无 | 10000 | SentencePiece | ✅ 兼容（架构相同） |
| InternLM2/3 | InternLM2ForCausalLM | 8 (GQA) | 有 | 1000000 | SentencePiece | ✅ 兼容（架构相同） |

> 任何使用 RMSNorm + GQA/MHA Attention + SwiGLU FFN 的 Dense Transformer 模型均可运行，只需正确的权重命名映射。

## 项目结构

```
hxinfer/
├── hxinfer-engine/            # 推理引擎核心
│   ├── include/
│   │   ├── base/              # Allocator、Buffer、Config、Dispatch
│   │   ├── tensor/            # Tensor 抽象层
│   │   ├── layer/             # Attention、Embedding、Linear、RMSNorm、SwiGLU、Transformer
│   │   ├── model/             # CausalLMModel（基类）、LlamaModel、Qwen2Model
│   │   ├── op/                # 算子声明（CPU/CUDA 自动调度）
│   │   └── cache/             # PrefixCacheManager
│   └── src/
│       ├── base/             # CPU/CUDA 分配器、Buffer
│       ├── tensor/           # Tensor 实现、CPU↔GPU 传输
│       ├── layer/            # 各层 forward 实现
│       ├── model/            # CausalLMModel forward 管线
│       └── op/
│           ├── cpu/          # CPU 算子（纯 C++）
│           └── cuda/         # 手写 CUDA kernel + cuBLAS
├── model_loaders/             # 模型加载器（独立模块）
│   ├── include/
│   │   ├── safetensors_loader.h     # 通用 safetensors 读取器
│   │   ├── llama_weight_loader.h    # LLaMA 权重组装器 → LlamaModel
│   │   └── qwen_weight_loader.h     # Qwen2 权重组装器 → Qwen2Model
│   └── src/
│       ├── safetensors_loader.cpp
│       ├── llama_weight_loader.cpp
│       └── qwen_weight_loader.cpp
├── tokenizer/                 # 分词器（独立模块）
│   ├── include/
│   │   ├── llama7b_tokenizer.h      # SentencePiece tokenizer
│   │   └── qwen_tokenizer.h         # Byte-level BPE tokenizer
│   └── src/
│       ├── llama7b_tokenizer.cpp
│       └── qwen_tokenizer.cpp
├── demos/                     # 演示程序 & Benchmark
├── main.cpp                   # 主入口
└── CMakeLists.txt
```

## 架构设计

### 引擎核心与加载器解耦

```
hxinfer-engine/   model_loaders/        tokenizer/
  ↓                  ↓                     ↓
推理计算            读 safetensors          文本编解码
forward()           组装模型对象            encode/decode
                    ↓
             ┌──────┴──────┐
        LlamaModel     Qwen2Model
        (LLaMA-2/3)   (Qwen2/2.5)
```

- `hxinfer-engine/` — 纯推理核心，定义 CausalLMModel 基类 + 各模型子类，不依赖加载器
- `model_loaders/` — 每个模型架构一个 weight loader，读取 safetensors + 组装模型
- `tokenizer/` — 每种分词方式一个实现，与引擎完全独立

### 模型类继承结构

```
CausalLMModel（基类：定义 Embedding → Transformer × N → Norm → LM Head 管线）
├── LlamaModel     （LLaMA-2/3, Mistral, InternLM2/3）
└── Qwen2Model     （Qwen2, Qwen2.5）
```

## 性能对比

### 1024/1024 端到端对比（单卡 H20，贪心解码，3 次平均）

| 指标 | hxinfer v1.0.0 | PyTorch (HF) | SGLang v0.5.2 |
|------|---------------:|-------------:|---------------:|
| **LLaMA-2-7B (FP16)** | | | |
| TTFT (ms) | 10,338 | 123 | 18.5 |
| ITL (ms) | 15.40 | 187.30 | 6.85 |
| Decode (tok/s) | 64.9 | 5.3 | 146.2 |
| 总吞吐 (tok/s) | 78.5 | 10.7 | 291.6 |
| **Qwen2.5-7B (FP16)** | | | |
| TTFT (ms) | 9,301 | 124 | 18.4 |
| ITL (ms) | 13.47 | 185.04 | 5.66 |
| Decode (tok/s) | 74.3 | 5.4 | 177.0 |
| 总吞吐 (tok/s) | 88.7 | 10.8 | 352.9 |

### 关键结论

- **Decode 阶段 hxinfer 远超 PyTorch** — 65.6 vs 5.3 tok/s（12×），因为 hxinfer 有增量 KV Cache，PyTorch 每步重算全部历史
- **Qwen2.5 比 LLaMA-2 快 13%** — GQA（4 KV heads vs 32）减少 KV cache 读写带宽，层数更少（28 vs 32）
- **与 SGLang 差距集中在** — 批量 prefill、CUDA Graph、FlashAttention、Continuous Batching

## 编译

### 依赖

- CMake >= 3.16
- 支持 C++17 的 GCC
- CUDA Toolkit >= 12.1
- SentencePiece、nlohmann-json

### 构建

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_CUDA_COMPILER=$(which nvcc)
cmake --build . -j$(nproc)
```

GPU 架构：Ampere (A100/3090) = `86`，Ada (RTX 4090) = `89`，Hopper (H20/H100) = `90`。

### 产出

- `libhxinfer_core.a` — 引擎核心静态库
- `libhxinfer_loaders.a` — 模型加载器静态库
- `libhxinfer_tokenizer.a` — 分词器静态库
- `hxinfer_engine` — 主推理程序

## 使用

### LLaMA-2-7B

```bash
export HXINFER_DATA_DIR=/path/to/llama2-7b-safetensors
./build/hxinfer_engine
```

### Qwen2.5-7B-Instruct

```bash
export HXINFER_DATA_DIR=/path/to/Qwen2.5-7B-Instruct
./build/demo_qwen
```

### Benchmark（1024 输入 / 1024 输出）

```bash
export HXINFER_DATA_DIR=/path/to/model
./build/bench_1024 1024 1024        # LLaMA-2-7B
./build/bench_1024_qwen 1024 1024  # Qwen2.5-7B
```

## 支持的算子

| 算子 | CPU | CUDA | 说明 |
|------|:---:|:----:|------|
| MatMul | ✓ | ✓ | cuBLAS FP16 GEMM |
| Attention | ✓ | ✓ | Multi-Head + GQA + KV Cache |
| RoPE | ✓ | ✓ | 旋转位置编码（支持 YaRN + 自定义 rope_theta）|
| RMSNorm | ✓ | ✓ | Warp 级规约 |
| SiLU | ✓ | ✓ | Sigmoid Linear Unit |
| SwiGLU | ✓ | ✓ | 融合 gate+silu+mul+down |
| Softmax | ✓ | ✓ | Safe softmax（减最大值防溢出）|
| Embedding | ✓ | ✓ | Token 嵌入查表 |
| Add / Mul | ✓ | ✓ | 逐元素（残差连接）|
| ArgMax | ✓ | ✓ | 贪心解码 |
| Sample | ✓ | ✓ | Temperature + Top-p 采样 |
| Bias Add | — | ✓ | Q/K/V bias（Qwen2.5）|

## License

MIT
