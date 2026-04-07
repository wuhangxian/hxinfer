import torch
import time

# 保持和 C++ 一致的数据规模: 1 * 4096 * 4096 = 16777216
size = 1 * 4096 * 4096
x = torch.randn(size, device='cuda', dtype=torch.float32)

# 预热 GPU
for _ in range(10):
    y = torch.nn.functional.silu(x)
torch.cuda.synchronize()

num_runs = 100
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
for _ in range(num_runs):
    y = torch.nn.functional.silu(x)
end_event.record()
torch.cuda.synchronize()

# time 换算为微秒(us)
avg_time_us = (start_event.elapsed_time(end_event) * 1000.0) / num_runs
print(f"PyTorch SiLU 平均耗时: {avg_time_us:.2f} us")

# 顺便测一下显存带宽利用率 (16M * 4 bytes 读 + 16M * 4 bytes 写 = 134 MB/次)
bytes_per_run = size * 4 * 2
bandwidth_gbps = (bytes_per_run / (avg_time_us / 1e6)) / (1024**3)
print(f"PyTorch 显存吞吐量: {bandwidth_gbps:.2f} GB/s")