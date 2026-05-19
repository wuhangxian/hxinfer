# hxinfer 推理引擎 —— 从 15M 到 7B 的完整升级文档

> 本文档面向 0 基础小白，详细记录了将 hxinfer 推理引擎从支持 Llama-15M 升级到支持 Llama-2-7B 的所有改动。

---

## 目录

1. [项目背景：为什么要升级？](#1-项目背景为什么要升级)
2. [FP16 混合精度推理](#2-fp16-混合精度推理)
3. [权重格式转换](#3-权重格式转换)
4. [SentencePiece 分词器](#4-sentencepiece-分词器)
5. [模型加载器](#5-模型加载器)
6. [显存管理修复](#6-显存管理修复)
7. [cuBLAS FP16 矩阵乘法](#7-cublas-fp16-矩阵乘法)
8. [FP16 Embedding 查表](#8-fp16-embedding-查表)
9. [Temperature + Top-p 采样](#9-temperature--top-p-采样)
10. [性能基准测试](#10-性能基准测试)
11. [文件变更清单](#11-文件变更清单)

---

## 1. 项目背景：为什么要升级？

### 原来的情况

hxinfer 最初只支持 **Llama-15M** 这个 tiny 模型。15M 模型只有 1500 万参数，权重文件约 60MB，FP32 精度就能轻松放进显存。但 15M 模型的输出质量很差，基本只能生成重复的、没有意义的话。

### 升级目标

让 hxinfer 支持 **Llama-2-7B**（67 亿参数）。7B 模型能生成有意义的英文文本，是真正的"能用"的模型。

### 核心挑战

| 挑战 | 15M 模型 | 7B 模型 |
|------|---------|---------|
| 参数量 | 15M | 6.7B（450 倍） |
| 权重大小 (FP32) | ~60MB | ~26GB |
| 权重大小 (FP16) | ~30MB | ~13.5GB |
| RTX 3090 显存 | 轻松放下 | FP32 放不下，必须用 FP16 |
| 分词器 | 自定义 tokenizer.bin | SentencePiece (.model) |
| 权重格式 | 自定义 binary | HuggingFace pytorch_model-*.bin |

**最关键的问题：** RTX 3090 只有 24GB 显存，7B 模型的 FP32 权重要 26GB，放不下！必须用 FP16 存储权重（13.5GB），才能装进显存。

---

## 2. FP16 混合精度推理

### 什么是 FP16？

- **FP32**：每个数用 32 位（4 字节）存储，精度高，占空间大
- **FP16**：每个数用 16 位（2 字节）存储，精度稍低，但占空间只有一半

7B 模型的权重用 FP16 存只需要 13.5GB，正好能放进 24GB 显存。

### 什么是"混合精度"？

```
权重（FP16，省显存） × 激活值（FP16，临时转换） → 输出（FP32，高精度）
```

- **权重**始终存在显存里，用 FP16 存省一半空间
- **计算时**把激活值也临时转成 FP16，让 Tensor Core 高速运算
- **输出**用 FP32，保证精度不丢失

### 代码改动：`include/base/config.h`

在 `DataType` 枚举里增加了 `kDataTypeFP16`：

```cpp
enum class DataType{
    kDataTypeUnknown=0,
    kDataTypeFP32=1,
    kDataTypeInt8=2,
    kDataTypeFP16=3,    // ← 新增！
};
```

同时在 `DataTypeSize` 里增加了 FP16 的大小返回：

```cpp
inline int DataTypeSize(DataType data_type){
    if(data_type==DataType::kDataTypeFP32) return sizeof(float);      // 4 字节
    else if(data_type==DataType::kDataTypeInt8) return sizeof(int8_t); // 1 字节
    else if(data_type==DataType::kDataTypeFP16) return sizeof(uint16_t); // 2 字节 ← 新增！
    else return 0;
}
```

**为什么改：** Tensor 类需要根据 DataType 计算内存大小。如果不加 FP16 支持，Tensor 分配的内存大小会算错（默认返回 0），导致程序崩溃。

---

## 3. 权重格式转换

### 问题

HuggingFace 的模型权重存在 `pytorch_model-00001-of-00002.bin` 等 Python pickle 文件里，C++ 无法直接读取。需要一个转换工具。

### 解决方案

写了 `tools/convert_weights.py` 脚本，把 Python pickle 格式转成 C++ 可读的二进制格式：

```
pytorch_model-*.bin  →  llama7b_weights.bin  +  llama7b_index.json
                          (13.5GB 连续二进制)     (48KB 索引文件)
```

### 转换过程

1. 读取 HuggingFace 的 `pytorch_model.bin.index.json`，获取所有权重名称和偏移
2. 逐个加载权重到内存
3. 如果权重是 bfloat16 格式，转成 float16（cuBLAS 不支持 bfloat16）
4. 把所有权重按固定顺序拼成一个大文件 `llama7b_weights.bin`
5. 生成 `llama7b_index.json`，记录每个权重在文件中的偏移量和大小

### 为什么这样设计？

- **一个大文件**：C++ 可以用 `mmap` 一次性映射整个文件到内存，比打开几百个小文件快得多
- **索引文件**：告诉 C++ 代码每个权重在大文件的哪个位置，直接 `seek` 过去读就行
- **FP16 存储**：bfloat16 和 float16 都是 16 位，但 GPU 的 Tensor Core 只支持 float16，所以必须转换

---

## 4. SentencePiece 分词器

### 问题

15M 模型用的是 Karpathy 自定义的 `tokenizer.bin` 格式。7B 模型用的是标准的 **SentencePiece** 格式（`tokenizer.model`），两者完全不兼容。

### 什么是分词器？

分词器把人类文字转成数字（token ID），模型才能处理。例如：

```
"Once upon a time" → [1, 15688, 1318, 263, 1123]
                      BOS  Once  upon  a   time
```

### 新增文件

#### `include/loader/llama7b_tokenizer.h`

```cpp
class Llama7BTokenizer{
private:
    sentencepiece::SentencePieceProcessor sp_;
public:
    explicit Llama7BTokenizer(const std::string& model_path);
    std::vector<int> encode(const std::string& text) const;  // 文字 → ID 列表
    std::string decode(int token_id) const;                    // ID → 文字
    int bos_id() const { return sp_.bos_id(); }               // 句子开始标记
    int eos_id() const { return sp_.eos_id(); }               // 句子结束标记
};
```

#### `src/loader/llama7b_tokenizer.cpp`

核心逻辑：

1. **`encode`**：调用 SentencePiece 库的 `Encode` 方法，把文本转成 token ID 列表
2. **`decode`**：调用 `IdToPiece` 把 token ID 转成字符串，然后做两个特殊处理：
   - **字节 token**：`<0x0A>` → 实际的换行符 `\n`（SentencePiece 用这种格式表示不可见字符）
   - **空格符号**：`▁`（Unicode U+2581）→ 真正的空格 ` `（SentencePiece 用这个特殊字符表示词首空格）

**为什么改：** 不同模型的分词器格式不同。7B 模型只能用 SentencePiece 解码，原来的自定义格式读不了。

---

## 5. 模型加载器

### 问题

15M 模型的加载器 `Llama15MLoader` 硬编码了 15M 的参数维度。7B 模型的维度完全不同（dim 从 288 变成 4096），需要一个新加载器。

### 关键参数对比

| 参数 | 15M | 7B |
|------|-----|-----|
| dim（隐藏维度） | 288 | 4096 |
| hidden_dim（FFN维度） | 768 | 11008 |
| n_layers（层数） | 6 | 32 |
| n_heads（注意力头数） | 6 | 32 |
| vocab_size（词表大小） | 32000 | 32000 |

### 新增文件

#### `include/loader/llama7b_loader.h` + `src/loader/llama7b_loader.cpp`

7B 加载器的工作流程：

1. **读取 `config.json`**：获取模型的维度参数（dim, hidden_dim, layers, heads 等）
2. **mmap 权重文件**：把 `llama7b_weights.bin` 映射到内存，不需要全部读进去
3. **读取索引**：从 `llama7b_index.json` 获取每个权重的偏移和大小
4. **按名称匹配权重**：HuggingFace 的命名规则是 `model.layers.N.self_attn.q_proj.weight`，需要映射到模型代码里对应的层
5. **分配显存并拷贝**：
   - Q/K/V/O 投影 + gate/up/down 投影：**保持 FP16** 直接拷贝到 GPU（省显存）
   - RMSNorm 权重：**FP16 转 FP32** 后拷贝到 GPU（只有 4096 个元素，FP32 才 16KB，无所谓）
   - Embedding 权重：**FP16 直接拷贝到 GPU**
   - lm_head 权重：**FP16 直接拷贝到 GPU**

**为什么这样分类：** 大权重（4096×4096 等）占显存大头，必须保持 FP16 才放得下。小权重（RMSNorm 只有 4096 个元素）转 FP32 没多少开销，反而能保证精度。

---

## 6. 显存管理修复

### 问题：所有输出都是零（最隐蔽的 bug）

在 7B 模型上首次推理时，所有 token 都被预测为 `<unk>`（unknown token，ID=0），因为**所有 logits 都是 0**。

### 根因分析

经过排查，发现是**内部张量的 device_type 设置错误**。

hxinfer 的算子分发（dispatch）机制是这样的：

```cpp
void matmul_tensor(input, weight, output){
    if(input->tensor_device_type() == DeviceType::kDeviceCUDA)
        matmul_cuda(input, weight, output);   // 走 GPU 路径
    else
        matmul_cpu(input, weight, output);    // 走 CPU 路径
}
```

问题出在：Attention 层、Transformer 层内部创建的临时张量（如 `curr_q_`, `k_cache_`, `v_cache_` 等），虽然分配器用的是 `CUDAAllocator`（内存确实在 GPU 上），但 `device_type` 没有被设置，默认值是 `DeviceType::kDeviceCPU`。

**后果：** 数据在 GPU 显存上，但算子以为在 CPU 上，调用了 CPU 版本的计算函数，对着 GPU 地址做 CPU 运算 → 全零输出。

### 修复方式

在所有创建内部张量的地方，手动设置 `device_type`：

**`include/layer/attention.h`**（关键改动）：

```cpp
// 之前：只创建了张量，没有设置 device_type
curr_q_ = std::make_shared<Tensor>(allocator, curr_shapes, DataType::kDataTypeFP32);

// 之后：显式设置 device_type
curr_q_ = std::make_shared<Tensor>(allocator, curr_shapes, DataType::kDataTypeFP32);
// ...
DeviceType dev = allocator->device_type();
curr_q_->tensor_set_device_type(dev);   // ← 新增！
curr_k_->tensor_set_device_type(dev);   // ← 新增！
curr_v_->tensor_set_device_type(dev);   // ← 新增！
k_cache_->tensor_set_device_type(dev);  // ← 新增！
v_cache_->tensor_set_device_type(dev);  // ← 新增！
after_qktv_->tensor_set_device_type(dev);  // ← 新增！
```

同样的修复也应用到了：
- **`include/layer/transformer.h`**：`norm_out_`, `attn_out_`, `ffn_out_` 三个张量
- **`include/model/llama_model.h`**：`ping_`, `pang_` 两个张量

**为什么改：** 这是整个 7B 升级过程中最关键的 bug 修复。没有这个修复，模型输出的 logits 全是 0，无论权重多正确都无法生成有意义的文字。

### 另一个关键改动：删除 `after_qkt_`

原来的 Attention 层有一个 `after_qkt_` 张量，用来存储 Q×K^T 的注意力分数矩阵。它的形状是 `{head, max_seq_len, max_seq_len}`：

- 15M 模型：`{6, 256, 256}` = 约 1.5MB × 6 层 = 9MB，没问题
- 7B 模型：`{32, 4096, 4096}` = 约 2GB × 32 层 = **64GB**，直接爆显存！

但仔细检查代码发现，**`after_qkt_` 在 forward 中根本没有被使用**（实际用的是 `attention_score_cuda` 函数里创建的临时张量）。所以直接删掉了。

---

## 7. cuBLAS FP16 矩阵乘法

### 问题

7B 模型的权重是 FP16 格式，但原来的 `matmul_cuda` 只支持 FP32×FP32。需要支持 FP16 权重的矩阵乘法。

### 第一次尝试（失败了）

直觉上，既然输入是 FP32、权重是 FP16，直接用 `cublasGemmEx` 指定混合类型不就行了吗？

```cpp
cublasGemmEx(handle,
    CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
    &alpha,
    d_B, CUDA_R_16F, K,    // 权重 FP16
    d_A, CUDA_R_32F, K,    // 输入 FP32  ← 问题在这！
    &beta,
    d_C, CUDA_R_32F, N,
    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

**结果：cuBLAS 静默返回全零！** 不报错，但输出全是 0。

**原因：** `cublasGemmEx` 不支持 A=FP32 + B=FP16 的混合类型组合。当 A 和 B 的类型不匹配时，cuBLAS 不报错，而是直接输出 0。这是一个**非常隐蔽的坑**。

### 最终方案

先把 FP32 输入转换成 FP16，再用 FP16×FP16→FP32 的路径：

```cpp
// 1) 启动 CUDA kernel，把 FP32 输入转成 FP16
fp32_to_fp16_kernel<<<blocks, threads>>>(d_A, ws, n_input);

// 2) 用 FP16 × FP16 → FP32（cuBLAS 完全支持）
cublasGemmEx(handle,
    CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
    &alpha,
    d_B, CUDA_R_16F, K,    // 权重 FP16
    ws,  CUDA_R_16F, K,    // 输入 FP16（刚转的）
    &beta,
    d_C, CUDA_R_32F, N,    // 输出 FP32（保持精度）
    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

### `src/op/cuda/matmul_cuda.cu` 完整改动

1. **新增 `fp32_to_fp16_kernel`**：CUDA kernel，每个线程把一个 float 转成 `__half`
2. **新增静态工作区**：用 `static __half* ws` 保存转换后的 FP16 输入，避免每次 `cudaMalloc`
3. **FP16 路径**：先转换输入，再调 `cublasGemmEx`
4. **FP32 路径**：保持原来的 `cublasSgemm`，15M 模型继续用

---

## 8. FP16 Embedding 查表

### 问题

7B 模型的 embedding 权重也是 FP16，原来的 embedding kernel 只支持 FP32 权重。

### `src/op/cuda/embedding_cuda.cu` 改动

新增 `embedding_kernel_fp16`：

```cpp
__global__ void embedding_kernel_fp16(
    const int* token_ids, const __half* weight,
    float* output, int dim, size_t num_tokens)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < num_tokens * dim){
        int token_id = token_ids[idx / dim];
        output[idx] = __half2float(weight[token_id * dim + idx % dim]);
        //              ^^^^^^^^^^^^^ 从 FP16 转 FP32 输出
    }
}
```

**逻辑：** 输入是 token ID（整数），权重是 FP16，输出是 FP32。查表时把 FP16 的 embedding 向量实时转成 FP32。

---

## 9. Temperature + Top-p 采样

### 问题

原来的推理使用 **argmax（贪心解码）**：每步选概率最大的那个 token。结果导致输出极度重复，例如：

> "She had a loving family, a loving family, and a loving family..."

### 为什么会重复？

语言模型在每一步给所有可能的 token 分配概率。贪心解码每次都选最"安全"的那个，陷入重复循环。就像一个人说话永远只说最有把握的那个词，最后就是说车轱辘话。

### 解决方案：Temperature + Top-p 采样

#### Temperature（温度）

Temperature 控制概率分布的"尖锐程度"：

```
概率 = exp(logit / temperature) / Σexp(logit / temperature)
```

- **temperature = 0**：等同于 argmax（只选最大概率的）
- **temperature = 0.6**：较保守，倾向于高概率 token
- **temperature = 0.8**：适中，有一定随机性
- **temperature = 1.0**：原始分布
- **temperature > 1.0**：更随机，更有"创意"但可能不连贯

#### Top-p（核采样）

Top-p 只从概率最高的那些 token 中采样，直到累积概率达到 p：

- **top_p = 0.9**：只从累积概率前 90% 的 token 中选，排除掉后面 10% 的"垃圾" token
- **top_p = 1.0**：从所有 token 中选（不做过滤）

### 代码实现

#### 新增 `src/op/sample_tensor.cpp`

```cpp
int sample_tensor(const std::shared_ptr<Tensor>& logits, float temperature, float top_p){
    // 1) 把 logits 从 GPU 拷贝到 CPU（32000 个 float，约 128KB，很快）
    // 2) Temperature scaling：每个 logit 除以 temperature
    // 3) Softmax：转成概率分布
    // 4) Top-p 过滤：按概率降序排列，累加到 top_p 就截断
    // 5) 重新归一化
    // 6) 按概率随机采样一个 token
}
```

**为什么在 CPU 上做采样？** 32000 个 logits 只有 128KB，从 GPU 拷到 CPU 只需几十微秒。在 CPU 上排序和采样比写 CUDA kernel 简单得多，而且采样本身不是性能瓶颈。

#### `demos/demo_gpu_llama7b.cpp` 改动

```cpp
// 之前：
next_token = argmax_tensor(logits);

// 之后：
next_token = sample_tensor(logits, temperature, top_p);
```

同时增加了命令行参数支持：

```bash
./demo_gpu_llama7b 0.8 0.9   # temperature=0.8, top_p=0.9
```

### 效果对比

| 解码方式 | 输出效果 |
|---------|---------|
| argmax (贪心) | "a loving family, a loving family, and a loving family..." (重复循环) |
| temperature=0.6, top_p=0.9 | "there lived a very kind and loving father. He had a little daughter..." (有变化但偶尔还是重复) |
| temperature=0.8, top_p=0.9 | "there was a prince who loved riding. He was a brave soldier..." (更自然) |

采样之后虽然不再死循环式重复，但文本质量仍然不高。这是因为我们使用的 **Yarn-Llama-2-7b-128k** 模型本身质量有限（它是在 Llama-2-7B 基座上做了 YaRN 长上下文微调，可能影响了生成质量），以及我们的 prefill 是逐 token 做的（fake prefill），和真正的 batched prefill 效果有差异。

---

## 10. 性能基准测试与优化

### RTX 3090 硬件参数

| 参数 | 值 |
|------|-----|
| FP16 Tensor Core 算力 | 142 TFLOPS |
| 显存带宽 | 936 GB/s |
| 显存容量 | 24 GB |

### 理论分析

Llama-2-7B 单步 decode（batch_size=1）的性能瓶颈分析：

#### 计算量

每步需要做 32 层 Transformer 的前向传播，每层有 7 个矩阵乘法（Q/K/V/O 投影 + gate/up/down FFN）。每个矩阵乘的计算量 = 2 × M × K × N，其中 M=1（decode 阶段只有一个 token）。

总计算量 ≈ 2 × 6.7B = **13.4 GFLOPs**

#### 受计算限制的理论 TPS

```
TPS_compute = 142 TFLOPS / 13.4 GFLOPs = 10,597 tok/s
```

#### 受带宽限制的理论 TPS

每步需要从显存读取所有权重（13.5GB FP16）：

```
TPS_bandwidth = 936 GB/s / 13.5 GB = 69.3 tok/s
```

#### 结论

**Decode 阶段是带宽瓶颈！** 因为 batch_size=1 时，计算量很小但还是要读取全部权重。实际 TPS 上限约 **69 tok/s**。

### 优化历程

#### 第一轮：初始版本（~45 tok/s）

直接把 15M 的架构搬到 7B，加上 FP16 权重支持。每步 decode 流程：

1. 前向传播（32 层 Transformer）
2. 采样：把 32000 个 logits (128KB) 从 GPU 拷到 CPU
3. CPU 上做 softmax + 排序 + top-p 采样
4. 把选中的 token ID 拷回 GPU 继续下一步

**瓶颈：** 步骤 2-3 中的 CPU 排序 32000 个元素要 ~870us，加上 D2H 拷贝 ~34us，合计约 1.9ms/step。纯模型推理只要 ~19.5ms/step，采样占了 ~10%！

#### 第二轮：cuBLAS 优化 + 融合 QKV/Gate+Up 投影（~51 tok/s，纯推理）

三个关键优化：

**1. cuBLAS 数学模式优化**

```cpp
cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
```

告诉 cuBLAS 优先使用 Tensor Core，即使有少量精度损失也可以。这让 FP16 GEMM 路径更高效。

**2. Q/K/V 共享 fp32→fp16 转换**

原来 Q、K、V 三个投影各自独立做 matmul，每次都要把 FP32 输入转成 FP16（3 次 kernel launch + 3 次转换）。

优化后：只做 **1 次** fp32→fp16 转换，然后 3 次 GEMM 共享同一个 FP16 输入。

```cpp
void matmul_qkv_cuda(input, wq, wk, wv, q_out, k_out, v_out){
    fp32_to_fp16(input, ws);  // 只转一次！
    cublasGemmEx(ws, wq, q_out);  // 共享 ws
    cublasGemmEx(ws, wk, k_out);  // 共享 ws
    cublasGemmEx(ws, wv, v_out);  // 共享 ws
}
```

**3. Gate+Up 共享 fp32→fp16 转换**

同样的思路：SwiGLU 中的 gate_proj 和 up_proj 共享输入，只转一次 FP16。

这些优化让**纯模型推理**从 ~45 提升到 ~51 tok/s（提升 13%）。

#### 第三轮：GPU 采样（~49.7 tok/s，含采样）

原来的采样流程（在 CPU 上做）：

```
GPU logits → D2H 拷贝 128KB → CPU softmax → CPU 排序 32K → CPU 采样 → 返回 token ID
                            ~34us          ~870us         ~几us
```

优化后：softmax + top-p + 采样全部在 **GPU** 上完成，只拷贝 **4 字节**（1 个 int）回 CPU：

```
GPU softmax → GPU top-p → GPU 采样 → D2H 拷贝 4B → 返回 token ID
   ~10us       ~5us       ~1us      ~5us
```

关键数据对比：

| 方式 | D2H 拷贝量 | 采样总耗时 |
|------|-----------|-----------|
| CPU 采样 | 128KB | ~1.9 ms/step |
| GPU 采样 | 4B | ~0.65 ms/step |

### 最终实测数据

| 框架 | TPS | 理论效率 |
|------|-----|---------|
| 理论带宽极限 | 69.3 tok/s | 100% |
| hxinfer 纯推理 (无采样) | **51.3 tok/s** | 74.0% |
| hxinfer + GPU 采样 | **49.7 tok/s** | 71.7% |
| hxinfer + argmax | **51.0 tok/s** | 73.6% |
| PyTorch (FP16) | **48.3 tok/s** | 69.6% |

### 分析

- **hxinfer 的纯推理速度（51.3 tok/s）已经超过 PyTorch（48.3 tok/s）！**
- 加上 GPU 采样后（49.7 tok/s），仍然比 PyTorch 快
- GPU 采样开销从 CPU 版的 1.9ms 降到 0.65ms，但仍有优化空间（主要受 PCIe 延迟限制）
- argmax 几乎零开销（GPU 端做 reduction，只需拷回 4 字节）
- 两者都没达到理论极限 69.3 tok/s，主要因为：
  1. cuBLAS GEMM 对 M=1 的小矩阵利用率低
  2. KV Cache 读写需要额外显存带宽
  3. RMSNorm、RoPE 等逐元素操作的 kernel launch 开销
- 进一步优化方向：Flash Attention、Continuous Batching、CUDA Graph

---

## 11. 文件变更清单

### 新增文件

| 文件 | 作用 |
|------|------|
| `include/loader/llama7b_loader.h` | 7B 模型加载器头文件 |
| `src/loader/llama7b_loader.cpp` | 7B 模型加载器实现（读 config.json + mmap 权重） |
| `include/loader/llama7b_tokenizer.h` | SentencePiece 分词器头文件 |
| `src/loader/llama7b_tokenizer.cpp` | 分词器实现（encode/decode + 特殊字符处理） |
| `demos/demo_gpu_llama7b.cpp` | 7B 模型推理 demo（含采样参数支持） |
| `src/op/cuda/sample_tensor.cu` | GPU 端 Temperature + Top-p 采样（softmax + 采样全在 GPU） | 替代 CPU 采样，减少 D2H 拷贝开销 |
| `tools/convert_weights.py` | HuggingFace 权重 → C++ 二进制格式转换脚本 |
| `tools/benchmark_pytorch.py` | PyTorch 7B 模型基准测试脚本 |

### 修改文件

| 文件 | 改了什么 | 为什么改 |
|------|---------|---------|
| `include/base/config.h` | 增加 `kDataTypeFP16=3` 和 `sizeof(uint16_t)` | Tensor 需要识别 FP16 类型来正确分配内存 |
| `src/op/cuda/matmul_cuda.cu` | 增加 FP16 权重路径 + cuBLAS Tensor Core 模式 + QKV/GateUp 融合 | 7B 权重是 FP16；融合减少 fp32→fp16 转换和 kernel launch 次数 |
| `src/op/cuda/embedding_cuda.cu` | 增加 `embedding_kernel_fp16` | 7B 的 embedding 权重是 FP16 |
| `include/layer/attention.h` | 删除 `after_qkt_`；所有内部张量加 `tensor_set_device_type(dev)` | 7B 的 after_qkt_ 会占 64GB；修复 device_type 导致的全零 bug |
| `include/layer/swiglu.h` | 新增 `enable_fused()` 方法 | 7B 时启用 Gate+Up 融合路径 |
| `include/layer/transformer.h` | norm_out_, attn_out_, ffn_out_ 加 `tensor_set_device_type(dev)`；7B 版构造函数加融合开关 | 修复 device_type bug；7B 支持 QKV/GateUp 融合 |
| `include/model/llama_model.h` | ping_, pang_ 加 `tensor_set_device_type(dev)`；新增 `get_blocks()` | 修复 device_type bug；暴露 blocks_ 给 Prefix Cache |
| `include/op/math_ops.h` | 新增 `sample_tensor`、`matmul_qkv_cuda`、`matmul_gate_up_cuda` 声明 | 支持 GPU 采样和融合 QKV/GateUp 投影 |
| `CMakeLists.txt` | nvcc 路径改为 /workspace/whx/cuda-12.1/bin/nvcc；架构改为 86；链接 sentencepiece | 适配服务器环境（CUDA 在 /workspace 下）和 RTX 3090（sm_86） |

---

## 附录：关键概念速查

### Q: 什么是 mmap？

A: `mmap`（memory-mapped file）是 Linux 系统调用，把文件直接映射到进程的虚拟地址空间。好处是：
- 不需要 `read()` 系统调用，直接用指针访问文件内容
- 操作系统按需加载，不用一次性把 13.5GB 全读进内存
- 多个进程可以共享同一份文件映射

### Q: 什么是 cuBLAS？

A: NVIDIA 提供的线性代数库，是 GPU 上做矩阵乘法的标准接口。我们用的函数：
- `cublasSgemm`：FP32 矩阵乘法
- `cublasGemmEx`：扩展版，支持 FP16/FP32 混合精度

### Q: 什么是 Tensor Core？

A: NVIDIA GPU 里的专用矩阵计算单元。RTX 3090 的第三代 Tensor Core 可以在一个时钟周期内完成 4×4×4 的 FP16 矩阵乘法。但 batch_size=1 时矩阵太小（1×4096），Tensor Core 利用率很低。

### Q: 为什么 decode 是带宽瓶颈而不是计算瓶颈？

A: 因为 batch_size=1 时，每次只处理 1 个 token，矩阵乘法的 M 维度只有 1。这意味着：
- 计算量极小：1×4096×4096 ≈ 16M 乘加 ≈ 0.03 GFLOPs
- 但还是要从显存读取整个 4096×4096 的权重矩阵（16MB FP16）
- GPU 算力远大于需要算的量，大部分时间在等数据从显存读出来
- 这就是"带宽瓶颈"：GPU 算得快但数据送得慢
