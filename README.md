# hxinfer

从零手写的轻量级 LLM 推理引擎，基于 C++/CUDA，支持 LLaMA 模型家族，CPU 和 GPU 双后端。

## 特性

- **LLaMA 模型支持** — LLaMA-2 15M（CPU）和 7B（GPU），支持 YaRN RoPE 扩展到 128K 上下文长度
- **CUDA 加速** — 全部算子均有手写 CUDA kernel：MatMul、Attention、RMSNorm、RoPE、SiLU、Embedding、Argmax 等
- **FP16 / FP32 双精度** — 7B 模型使用 FP16 激活值（与 PyTorch 行为一致），15M 使用 FP32
- **算子融合** — 融合 QKV 投影、融合 Gate+Up 投影、融合 SwiGLU（gate+silu+mul+down）
- **前缀缓存（Prefix Cache）** — KV Cache 快照与恢复，相同前缀 token 跳过重复 forward 计算
- **Temperature + Top-p 采样** — 除贪心 argmax 外，支持可配置的温度采样
- **SentencePiece 分词器** — 原生 C++ 实现 LLaMA 分词

## 项目结构

```
hxinfer/
├── include/
│   ├── base/          # 内存分配器、Buffer、配置、调度
│   ├── tensor/        # Tensor 抽象层（CPU/CUDA）
│   ├── layer/         # Embedding、Linear、RMSNorm、Attention、SwiGLU、Transformer
│   ├── model/         # LlamaModel
│   ├── loader/        # Llama15MLoader、Llama7BLoader、分词器
│   ├── op/            # 算子声明（CPU/CUDA 自动调度）
│   └── cache/         # PrefixCacheManager
├── src/
│   ├── base/          # CPU 分配器、CUDA 分配器
│   ├── tensor/        # Tensor 实现
│   ├── layer/         # 各层 forward 实现
│   ├── model/         # LlamaModel forward
│   ├── loader/        # 权重加载 & 分词
│   └── op/
│       └── cuda/      # 手写 CUDA kernel
├── demos/             # 逐步演示 & 性能测试
├── tools/
│   ├── convert_weights.py   # HuggingFace 权重 → hxinfer 二进制格式
│   └── benchmark_pytorch.py # PyTorch 基准对比
└── docs/
```

## 编译

### 依赖

- CMake >= 3.16
- CUDA Toolkit >= 12.1
- 支持 C++17 的 GCC
- SentencePiece（分词器用）

### 构建

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_COMPILER=$(which nvcc) -DCMAKE_CUDA_ARCHITECTURES="86"
cmake --build . -j$(nproc)
```

产出：
- `hxinfer_engine` — 主推理程序
- `demos/` 下的各独立演示程序

## 使用

### 7B GPU 推理

```bash
# 准备权重（只需转换一次）
python tools/convert_weights.py \
  --model_dir /path/to/huggingface/Yarn-Llama-2-7b-128k \
  --output_dir /path/to/hxinfer-data

# 运行推理
./build/demo_gpu_llama7b                        # 默认采样
./build/demo_gpu_llama7b --greedy               # 贪心解码
./build/demo_gpu_llama7b 0.8 0.9                # temperature=0.8, top_p=0.9
```

### 15M CPU 推理

```bash
./build/20_model_llama_infer
```

### 前缀缓存演示

```bash
./build/demo_prefix_cache
```

### 权重转换

```bash
python tools/convert_weights.py \
  --model_dir /path/to/huggingface/model \
  --output_dir /path/to/output
```

产出：
- `llama7b_weights.bin` — 所有权重连续存储的 FP16 字节流
- `llama7b_index.json` — 每个权重张量的偏移量/形状/数据类型索引

## CUDA 算子

| 算子 | 实现方式 | 说明 |
|------|---------|------|
| MatMul | cuBLAS | 7B 使用 FP16 Tensor Core |
| Attention | 自定义 Kernel | 融合 KV Cache 更新 + 得分计算 + 加权求和 |
| RMSNorm | 自定义 Kernel | Warp 级归约 |
| RoPE | 自定义 Kernel | 支持 YaRN 缩放 |
| SiLU | 自定义 Kernel | 逐元素，比 PyTorch 快约 0.3% |
| Embedding | 自定义 Kernel | Token ID 查表 |
| Add / Mul | 自定义 Kernel | 残差连接与缩放 |
| Argmax | 自定义 Kernel | 贪心解码 |
| Sample | 自定义 Kernel | Temperature + Top-p 采样 |

## 前缀缓存（Prefix Cache）

`PrefixCacheManager` 支持共享前缀 token 的 KV Cache 复用：

1. **Store** — 将前缀 token 对应的 KV Cache 快照存入哈希表（以 token ID 序列的 FNV-1a 哈希为 key）
2. **Lookup** — O(1) 哈希查找，碰撞时精确比对 token 序列
3. **Restore** — 将缓存的 KV 快照 memcpy 回模型的 KV Cache，跳过前缀部分的 forward 计算

适用于多个请求共享相同 system prompt 或对话历史的场景。

## 性能

RTX 3090 上 LLaMA-2-7B decode 速度：

| 引擎 | 速度 | 硬件利用率 |
|------|------|-----------|
| PyTorch (HF) | ~38 tok/s | 基准 |
| hxinfer | ~51 tok/s | 带宽上限的 74% |
| 理论带宽极限 | ~69 tok/s | 100%（936 GB/s） |

## 开发历程

项目从最底层的内存管理开始，逐步构建：
1. 原始内存管理 → 分配器 → Tensor 抽象
2. 逐个实现算子（Add、Mul、MatMul、RMSNorm、RoPE、SiLU、Embedding、Argmax）
3. 组装层（Linear、Embedding、Attention、SwiGLU、Transformer）
4. 完成模型（LlamaModel）和权重加载
5. CPU 推理跑通 → CUDA 算子逐个适配 → GPU 推理
6. 算子融合优化、FP16 支持、7B 模型适配
7. Prefix Cache 功能

`demos/` 目录下的编号文件记录了完整的开发过程。

## License

MIT
