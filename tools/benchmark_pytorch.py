"""
PyTorch 基准测试：Llama-2-7B 单步 decode 速度
与 hxinfer C++ 推理引擎做对比
"""
import torch
import time
import json
import os

DATA_DIR = "/workspace/models/Yarn-Llama-2-7b-128k"

def main():
    print("=" * 60)
    print("PyTorch Llama-2-7B Decode 基准测试")
    print("=" * 60)

    # 1. 加载模型（FP16，直接放 GPU）
    print("\n[1/3] 加载模型...")
    from transformers import LlamaForCausalLM, LlamaConfig
    import sys

    # 抑制 yarn 配置警告
    import warnings
    warnings.filterwarnings("ignore")

    config = LlamaConfig.from_pretrained(DATA_DIR)
    print(f"  模型配置: dim={config.hidden_size}, layers={config.num_hidden_layers}, "
          f"heads={config.num_attention_heads}, vocab={config.vocab_size}")

    model = LlamaForCausalLM.from_pretrained(
        DATA_DIR,
        torch_dtype=torch.float16,
    ).cuda()
    model.eval()
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # 2. 准备 KV Cache（模拟 decode 阶段）
    print("\n[2/3] 准备 KV Cache...")
    seq_len = 1  # decode 阶段每次只处理 1 个 token
    input_ids = torch.tensor([[1]], dtype=torch.long, device="cuda")  # BOS token

    # Warm up: 先跑一次 prefill 让 KV Cache 初始化
    with torch.no_grad():
        _ = model(input_ids, use_cache=True)

    # 3. Benchmark: 连续 decode 多步
    print("\n[3/3] 基准测试 (单步 decode, batch_size=1)...")
    num_warmup = 5
    num_steps = 50

    # 先做 warmup
    past_key_values = None
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        past_key_values = out.past_key_values

    for _ in range(num_warmup):
        with torch.no_grad():
            out = model(input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            next_token = out.logits.argmax(dim=-1)
            input_ids = next_token

    # 正式测试
    torch.cuda.synchronize()
    t_start = time.time()

    with torch.no_grad():
        for _ in range(num_steps):
            out = model(input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            next_token = out.logits.argmax(dim=-1)
            input_ids = next_token

    torch.cuda.synchronize()
    t_end = time.time()

    elapsed = t_end - t_start
    tps = num_steps / elapsed

    print(f"\n{'=' * 60}")
    print(f"PyTorch 单步 Decode 结果:")
    print(f"  步数: {num_steps}")
    print(f"  总耗时: {elapsed:.3f} s")
    print(f"  平均每步: {elapsed / num_steps * 1000:.2f} ms")
    print(f"  速度: {tps:.2f} tok/s")
    print(f"{'=' * 60}")

    # 4. 理论峰值计算
    print("\n理论峰值计算 (RTX 3090):")
    print("  FP16 Tensor Core 峰值: 142 TFLOPS (FP16)")
    print("  显存带宽: 936 GB/s")

    # Llama-2-7B 单步 decode 的计算量
    # 每层: QKV投影 + O投影 + gate/up/down 投影 = 7 个矩阵乘法
    # 每个矩阵乘: 2 * M * K * N FLOPs (M=1 for decode)
    # 总 FLOPs ≈ 2 * 7B_params (每步) ≈ 14G FLOPs
    flops_per_step = 2 * 6.7e9  # 2 * 6.7B params
    compute_tps = 142e12 / flops_per_step  # 受计算限制的理论 TPS

    # 受带宽限制: 每步需要读取所有权重 (13.5GB FP16)
    weight_size_gb = 13.5
    bandwidth_tps = 936 / weight_size_gb  # 受带宽限制的理论 TPS

    print(f"\n  每步计算量: {flops_per_step / 1e9:.1f} GFLOPs")
    print(f"  计算限制理论 TPS: {compute_tps:.1f} tok/s")
    print(f"  带宽限制理论 TPS: {bandwidth_tps:.1f} tok/s")
    print(f"  (Decode 是带宽瓶颈，理论最大 ≈ {bandwidth_tps:.1f} tok/s)")

    # 5. 对比总结
    print(f"\n{'=' * 60}")
    print(f"速度对比总结:")
    print(f"  理论带宽极限:  {bandwidth_tps:.1f} tok/s")
    print(f"  PyTorch 实测:  {tps:.1f} tok/s  (效率: {tps/bandwidth_tps*100:.1f}%)")
    print(f"  hxinfer 实测:  ~51.3 tok/s  (效率: {51.3/bandwidth_tps*100:.1f}%)")
    print(f"  hxinfer/PyTorch: {51.3/tps*100:.1f}%")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
