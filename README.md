# hxinfer - 从零构建的 C++/CUDA LLM 推理引擎

从零手写的轻量级 LLM 推理引擎，基于 C++/CUDA，支持 LLaMA 模型家族，CPU 和 GPU 双后端。纯 C++/CUDA 实现，零第三方深度学习框架依赖。

## 特性

- **LLaMA 模型支持** — LLaMA-2 7B（GPU），支持 YaRN RoPE 扩展到 128K 上下文长度
- **双后端支持** — CPU（纯 C++）和 CUDA（手写 Kernel + cuBLAS），运行时 Zero-Cost Dispatch
- **FP16 推理** — 使用 FP16 激活值，与 PyTorch 行为一致
- **零框架依赖** — 不依赖 PyTorch/TensorFlow，仅需 C++17 编译器 + CUDA Toolkit
- **算子融合** — 融合 QKV 投影、融合 Gate+Up 投影、融合 SwiGLU（gate+silu+mul+down）
- **mmap 零拷贝加载** — 模型权重通过内存映射直接加载，避免冗余拷贝
- **KV Cache** — 自回归解码时缓存历史 K/V，避免重复计算
- **前缀缓存（Prefix Cache）** — KV Cache 快照与恢复，相同前缀 token 跳过重复 forward 计算，大幅加速多轮对话
- **Temperature + Top-p 采样** — 除贪心 argmax 外，支持可配置的温度采样
- **SentencePiece 分词器** — 原生 C++ 实现 LLaMA 分词

## 项目结构

```
hxinfer/
├── include/
│   ├── base/          # 内存分配器、Buffer、配置、类型分发
│   ├── tensor/        # Tensor 抽象层（CPU/CUDA 切换、reshape、类型安全访问）
│   ├── layer/         # Embedding、Linear、RMSNorm、Attention、SwiGLU、Transformer
│   ├── model/         # LlamaModel 顶层定义
│   ├── loader/        # Llama7BLoader、分词器
│   ├── op/            # 算子声明（CPU/CUDA 自动调度）
│   └── cache/         # PrefixCacheManager（前缀缓存加速）
├── src/
│   ├── base/          # CPU 分配器、CUDA 分配器
│   ├── tensor/        # Tensor 实现、CPU↔GPU 数据传输
│   ├── layer/         # 各层 forward 实现
│   ├── model/         # LlamaModel forward 管线
│   ├── loader/        # mmap 零拷贝加载 & 分词
│   └── op/
│       ├── cpu/       # CPU 算子实现（纯 C++，无任何依赖）
│       └── cuda/      # 手写 CUDA kernel + cuBLAS
├── demos/             # 逐步演示 & 性能测试（从 Hello World 到完整推理）
├── tools/
│   ├── convert_weights.py   # HuggingFace 权重 → hxinfer 二进制格式
│   └── benchmark_pytorch.py # PyTorch 基准对比
├── main.cpp           # 主入口：自回归文本生成
└── CMakeLists.txt     # CMake 构建配置
```

## 编译

### 依赖

- CMake >= 3.16
- 支持 C++17 的 GCC
- CUDA Toolkit >= 12.1（GPU 推理需要）
- SentencePiece（分词器用）、nlohmann-json（权重索引解析用）

### 构建

```bash
mkdir -p build && cd build
# nvcc 默认从 PATH 自动探测；GPU 架构默认 86 (Ampere)。
# 按需覆盖： -DCMAKE_CUDA_COMPILER=$(which nvcc) -DCMAKE_CUDA_ARCHITECTURES=90
cmake ..
cmake --build . -j$(nproc)
```

常见 GPU 架构：Ampere (A100/3090) = `86`，Ada (RTX 4090) = `89`，Hopper (H20/H100) = `90`。

产出：
- `libhxinfer_core.a` — 核心推理引擎静态库
- `hxinfer_engine` — 主推理程序
- `demos/` 下的各独立演示程序

## 使用

### 7B GPU 推理

```bash
# 1. 准备权重（只需转换一次）
python tools/convert_weights.py \
  --model_dir /path/to/huggingface/Yarn-Llama-2-7b-128k \
  --output_dir /path/to/hxinfer-data

# 2. 指定数据目录并运行（默认目录可用环境变量覆盖）
export HXINFER_DATA_DIR=/path/to/hxinfer-data
./build/demo_gpu_llama7b                        # 默认采样
./build/demo_gpu_llama7b --greedy               # 贪心解码
./build/demo_gpu_llama7b 0.8 0.9                # temperature=0.8, top_p=0.9
```

### 前缀缓存演示

```bash
./build/demo_prefix_cache
```

### 权重转换产出

`tools/convert_weights.py` 将 HuggingFace 的 pickle 权重转换为 C++ 可 mmap 读取的格式：
- `llama7b_weights.bin` — 所有权重连续存储的 FP16 字节流
- `llama7b_index.json` — 每个权重张量的偏移量/形状/数据类型索引

## 支持的算子

| 算子 | CPU | CUDA | 说明 |
|------|:---:|:----:|------|
| MatMul | ✓ | ✓ | 矩阵乘法（CUDA 用 cuBLAS，7B 走 FP16 Tensor Core） |
| Attention | ✓ | ✓ | 多头注意力（融合 KV Cache 更新 + 得分 + 加权求和） |
| RoPE | ✓ | ✓ | 旋转位置编码（支持 YaRN 缩放） |
| RMSNorm | ✓ | ✓ | Root Mean Square 归一化（Warp 级归约） |
| SiLU | ✓ | ✓ | Sigmoid Linear Unit 激活 |
| SwiGLU | ✓ | ✓ | SiLU 门控线性单元（gate+silu+mul+down 融合） |
| Softmax | ✓ | ✓ | 带 safe-softmax（减最大值防溢出） |
| Embedding | ✓ | ✓ | Token 嵌入查表 |
| Add / Mul | ✓ | ✓ | 逐元素相加/相乘（残差连接与缩放） |
| ArgMax | ✓ | ✓ | 概率最高 token 索引查找（贪心解码） |
| Sample | ✓ | ✓ | Temperature + Top-p 采样 |

## 设计亮点

### 1. 类型分发宏（Dispatch）
通过 `HXINFER_DISPATCH_ALL_TYPES` 宏在编译期完成 DataType 到具体 C++ 类型的映射，避免运行时开销：

```cpp
HXINFER_DISPATCH_ALL_TYPES(data_type, "matmul", {
    // 在此作用域内，scalar_t = float / int8_t ...
});
```

### 2. CPU/CUDA 双后端自动路由
每个算子的 `*_tensor()` 调度函数自动检测 Tensor 所在设备，分发到 CPU 或 CUDA 实现：

```
rmsnorm_tensor() → rmsnorm_cpu() / rmsnorm_cuda()
```

### 3. 内存分配器抽象
`CPUAllocator` / `CUDAAllocator` 提供统一的内存接口，Tensor 通过 Allocator 申请内存，天然支持 CPU/GPU 透明切换。

### 4. Transformer Block 流水线
每层 Transformer 按照 LLaMA 架构精确实现：
```
Input → RMSNorm → Attention(+Residual) → RMSNorm → SwiGLU(+Residual) → Output
```

## 前缀缓存（Prefix Cache）

`PrefixCacheManager` 支持共享前缀 token 的 KV Cache 复用：

1. **Store** — 将前缀 token 对应的 KV Cache 快照存入哈希表（以 token ID 序列的 FNV-1a 哈希为 key）
2. **Lookup** — O(1) 哈希查找，碰撞时精确比对 token 序列
3. **Restore** — 将缓存的 KV 快照 memcpy 回模型的 KV Cache，跳过前缀部分的 forward 计算

适用于多个请求共享相同 system prompt 或对话历史的场景。

## 性能

### RTX 3090 上 LLaMA-2-7B decode 速度

| 引擎 | 速度 | 硬件利用率 |
|------|------|-----------|
| PyTorch (HF) | ~38 tok/s | 基准 |
| hxinfer | ~51 tok/s | 带宽上限的 74% |
| 理论带宽极限 | ~69 tok/s | 100%（936 GB/s） |

- **自定义 SiLU Kernel**：比 PyTorch 原生实现略快约 0.3%
- **cuBLAS MatMul**：矩阵乘法使用 cuBLAS 加速，替换了初始的手写 naive kernel

### 与 sglang 对比（单卡 H20，LLaMA-2-7B，输入 1024 / 输出 1024）

在同一张 H20（HBM3 带宽约 4.0 TB/s）上，用相同的 1024 输入 / 1024 输出、batch=1、贪心解码做端到端对比。sglang 用 `bench_serving`（`--random-input-len 1024 --random-output-len 1024`），hxinfer 用对应的自定义基准。

| 指标 | hxinfer (batch=1) | sglang (并发=1) | 说明 |
|------|------:|------:|------|
| Decode 吞吐 (tok/s) | 34.5 | 154.4 | sglang 约 **4.5×** |
| 单 token 延迟 ITL (ms) | 29.0 | 6.4 | 越低越好 |
| 首 token 延迟 TTFT (ms) | 12001 | 69.5 | sglang 约 **173×**，见下 |
| 理论带宽极限 (tok/s) | \~301 | \~301 | 同卡同模型，两者共享此上限 |

**说明与差距来源：**

- **Decode（约 4.5× 差距）**：hxinfer 单 token 走 batch=1 的朴素 attention + 逐算子调用，没有 CUDA Graph 消除 kernel launch 开销，也没有 FlashAttention 类的访存优化，34.5 tok/s 仅到理论带宽极限（约 301 tok/s）的 **11%**。sglang 的 154 tok/s 也远未到单请求上限——单请求场景两者都受制于 kernel 调度与访存效率，而非算力。
- **TTFT（约 173× 差距，最关键的架构弱点）**：hxinfer 的 prefill 是**逐 token 串行**过模型（1024 个 token 一个一个 forward），所以首 token 要等约 12 秒；sglang 把整段 prompt 作为一个批次一次性并行 prefill，只需约 70ms。这是 hxinfer 当前最大的架构短板。
- **并发吞吐**：sglang 靠 continuous batching + PagedAttention，在高并发（约 291 路）下总吞吐可达约 **2900 tok/s**；hxinfer 目前只支持 batch=1，没有连续批处理，无法在这个维度扩展。

**结论**：hxinfer 是一个从零手写、零框架依赖的**教学/研究型**推理引擎，单请求 decode 已能到理论带宽的一成、正确复现 LLaMA-2-7B 的完整前向；但相比 sglang 这类生产级引擎，差距集中在三处工程优化上——**批量 prefill、CUDA Graph、continuous batching**，这也是后续最值得补齐的方向。

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
