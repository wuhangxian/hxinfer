# HXInfer - 从零构建的 C++/CUDA LLM 推理引擎

一个完全从零手写的 LLM 推理引擎，支持 LLaMA 架构模型在 CPU 和 NVIDIA GPU 上的高效推理。纯 C++/CUDA 实现，零第三方深度学习框架依赖。

## 架构概览

```
hxinfer/
├── include/
│   ├── base/           # 基础组件：Allocator、Buffer、Tensor、类型分发
│   ├── tensor/         # 张量抽象（CPU/CUDA 切换、reshape、类型安全访问）
│   ├── op/             # 数学算子接口（CPU + CUDA 双后端）
│   ├── layer/          # 神经网络层（Embedding、Attention、SwiGLU、Transformer 等）
│   ├── model/          # LLaMA 模型顶层定义
│   ├── loader/         # 模型权重加载 & Tokenizer
│   └── cache/          # Prefix Cache（前缀缓存加速）
├── src/
│   ├── base/           # Allocator、Buffer 实现（CPU: new/delete, CUDA: cudaMalloc/cudaFree）
│   ├── tensor/         # Tensor 实现、CPU↔GPU 数据传输
│   ├── op/             # 算子调度层（根据 DeviceType 分发 CPU/CUDA）
│   │   ├── cpu/        # CPU 算子实现（纯 C++，无任何依赖）
│   │   └── cuda/       # CUDA 算子实现（手写 Kernel + cuBLAS）
│   ├── layer/          # 层前向传播实现
│   ├── model/          # LLaMA 模型 forward 管线
│   └── loader/         # mmap 零拷贝加载 & Tokenizer
├── demos/              # 演示/测试用例（逐步演进的算子验证 demo）
├── models/             # 模型权重文件
├── main.cpp            # 主入口：自回归文本生成
└── CMakeLists.txt      # CMake 构建配置
```

## 特点

- **双后端支持**：CPU（纯 C++）和 CUDA（手写 Kernel + cuBLAS），运行时 Zero-Cost Dispatch
- **零框架依赖**：不依赖 PyTorch/TensorFlow，仅需 C++17 编译器 + CUDA Toolkit
- **mmap 零拷贝加载**：模型权重通过内存映射直接加载，避免冗余拷贝
- **权重共享**：Embedding 和 LM Head 权重自动共享，节省内存
- **KV Cache**：自回归解码时缓存历史 K/V，避免重复计算
- **Prefix Cache**：缓存相同前缀的完整 KV 快照，支持 O(1) 恢复，大幅加速多轮对话
- **贪心解码**：基于 argmax 的自回归 token 生成

## 快速开始

### 环境要求

- CMake >= 3.16
- GCC 支持 C++17
- CUDA Toolkit 12.1+（GPU 推理需要，推荐 SM 89 架构如 RTX 4090）

### 构建

```bash
cd hxinfer
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

构建产物：
- `libhxinfer_core.a` — 核心推理引擎静态库
- `hxinfer_engine` — 主可执行文件（LLaMA 故事生成 demo）

### 运行

```bash
# 确保 models/ 目录下有模型文件
./build/hxinfer_engine
```

程序会加载 `models/stories15M.bin`（一个 15M 参数的微型 LLaMA 模型，用于概念验证），进行自回归文本生成，输出 200 个 token 的童话故事。

### 准备模型

```bash
# stories15M 演示模型（约 60MB）
# 将其放置于 hxinfer/models/ 目录下

# LLaMA-2-7B（需要从 modelscope 下载，路径位于 ../models/modelscope/Llama-2-7b-ms/）
```

## 支持的算子

| 算子 | CPU | CUDA | 说明 |
|------|:---:|:----:|------|
| MatMul | ✓ | ✓ | 矩阵乘法（CUDA 使用 cuBLAS） |
| Attention | ✓ | ✓ | 多头注意力（含 KV Cache） |
| RoPE | ✓ | ✓ | 旋转位置编码 |
| RMSNorm | ✓ | ✓ | Root Mean Square 归一化 |
| SiLU | ✓ | ✓ | Sigmoid Linear Unit 激活 |
| SwiGLU | ✓ | ✓ | SiLU 门控线性单元 |
| Softmax | ✓ | ✓ | 带 safe-softmax（减最大值防溢出） |
| Embedding | ✓ | ✓ | Token 嵌入查表 |
| Add | ✓ | ✓ | 逐元素相加 |
| Mul | ✓ | ✓ | 逐元素相乘 |
| ArgMax | ✓ | ✓ | 概率最高 token 索引查找 |

## Demos 目录

`demos/` 下包含 24 个渐进式演示示例，记录了从"Hello World"到完整 LLaMA 推理的实现过程：

| # | 文件 | 内容 |
|---|------|------|
| 01 | `cuda_hello_world.cu` | CUDA 入门 |
| 02 | `allocator.cpp` / `allocator_test.cu` | 内存分配器（CPU/CUDA） |
| 03 | `buffer.cpp` | Buffer 缓冲区抽象 |
| 04 | `tensor.cpp` | Tensor 张量系统 |
| 05 | `naive_layer.cpp` / `relulayer_retry1.cpp` | 层抽象 & ReLU 层 |
| 06 | `linear.cpp` / `linearlayer-retry1/2.cpp` | 线性层 |
| 07 | `embedding.cpp` / `embedding_retry1.cpp` | Embedding 层 |
| 08 | `softmax_task.cpp` | Softmax 算子 |
| 09 | `rmsnorm_task.cpp` / `rmsnorm_retry1.cpp` | RMSNorm 算子 |
| 10 | `silu_task.cpp` | SiLU 激活算子 |
| 11 | `mul_task.cpp` | 逐元素乘法 |
| 12 | `rope_task.cpp` | RoPE 位置编码 |
| 13 | `matmul_task.cpp` | 矩阵乘法 |
| 14 | `add_task.cpp` | 逐元素加法 |
| 15 | `argmax_task.cpp` | ArgMax 算子 |
| 16 | `swiglu_block.cpp` | SwiGLU FFN 模块 |
| 17 | `naive_attention_block.cpp` / `retry1.cpp` | 注意力模块 |
| 18 | `llama_inference_front.cpp` | 推理前端入口 |
| 19 | `llama_infer.cpp` / `spilt1.cpp` / `bin_abstract_3.cpp` | LLaMA 推理主循环 |
| 20 | `model_llama_infer.cpp` | 完整模型推理 |
| 21 | `silu_cuda_test.cu` | SiLU CUDA 精度测试 |
| 22 | `matmul_cuda_test.cpp` | MatMul CUDA 测试 |
| 23 | `addmul_cuda_test.cpp` | AddMul CUDA 测试 |
| 24 | `silu_benchmark.cu` | SiLU 性能基准测试 |
| — | `llama_infer_cpu_v1/v2.cpp` | CPU 推理优化版本 |
| — | `demo_gpu_llama.cpp` | GPU 端到端推理 |
| — | `demo_prefix_cache.cpp` | Prefix Cache 演示 |

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

## 性能

- **CUDA 加速比**：GPU 推理速度约为 CPU 的 **4 倍**（stories15M 模型，RTX 4090 vs 单核 CPU）
- **自定义 SiLU Kernel**：比 PyTorch 原生实现略快约 0.3%
- **cuBLAS MatMul**：矩阵乘法使用 cuBLAS 加速，替换了初始的手写 naive kernel

## 许可证

MIT License
