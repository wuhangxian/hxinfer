import struct
import numpy as np
import torch
import torch.nn.functional as F
import math
import time
import sys

print("====== 🚀 正在启动 PyTorch 原生对照组引擎 ======")

# 1. 读取 tokenizer (带终极容错)
vocab = []
try:
    with open('/home/whx/hxinfer/models/tokenizer.bin', 'rb') as f:
        max_token_length = struct.unpack('i', f.read(4))[0]
        for _ in range(32000):
            score = struct.unpack('f', f.read(4))[0]
            length = struct.unpack('i', f.read(4))[0]
            word_bytes = f.read(length)

            # 🚀 极其强壮的解码，遇到生僻字直接变成 "?"，绝不崩溃！
            try:
                text = word_bytes.decode('utf-8')
            except UnicodeDecodeError:
                text = "?"

            if text.startswith(' '):
                text = ' ' + text[1:]
            elif text == '<0x0A>':
                text = '\n'
            vocab.append(text)
except Exception as e:
    print(f"❌ 读取 tokenizer 失败: {e}")
    sys.exit(1)

# 2. 读取模型权重
try:
    with open('/home/whx/hxinfer/models/stories15M.bin', 'rb') as f:
        header = f.read(28)
        dim, hidden_dim, layer, head, kv_head, vocab_size, seq_len = struct.unpack('iiiiiii', header)
        weights_data = np.frombuffer(f.read(), dtype=np.float32)
except Exception as e:
    print(f"❌ 读取模型权重失败: {e}")
    sys.exit(1)

print(f"--- 成功解析模型配置 ---")
print(f"Dim: {dim}, Hidden_dim: {hidden_dim}, Layers: {layer}, Heads: {head}, Vocab: {vocab_size}")

# 限制单线程，不写 set_grad_enabled(False) 防止版本报错
torch.set_num_threads(1)

# 3. 切割二进制权重
offset = 0
def get_tensor(shape):
    global offset
    size = np.prod(shape)
    w = weights_data[offset:offset+size].reshape(shape)
    offset += size
    return torch.from_numpy(w).clone()

embed_w = get_tensor((vocab_size, dim))
attn_norm_w = [get_tensor((dim,)) for _ in range(layer)]
wq_w = [get_tensor((dim, dim)) for _ in range(layer)]
wk_w = [get_tensor((dim, dim)) for _ in range(layer)]
wv_w = [get_tensor((dim, dim)) for _ in range(layer)]
wo_w = [get_tensor((dim, dim)) for _ in range(layer)]
ffn_norm_w = [get_tensor((dim,)) for _ in range(layer)]
w1_w = [get_tensor((hidden_dim, dim)) for _ in range(layer)]
w2_w = [get_tensor((dim, hidden_dim)) for _ in range(layer)]
w3_w = [get_tensor((hidden_dim, dim)) for _ in range(layer)]
final_norm_w = get_tensor((dim,))

head_dim = dim // head
offset += seq_len * head_dim

if offset < len(weights_data):
    lm_head_w = get_tensor((vocab_size, dim))
else:
    lm_head_w = embed_w

# 4. 推理核心逻辑
def rmsnorm(x, w, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

k_cache = [torch.zeros((seq_len, head, head_dim)) for _ in range(layer)]
v_cache = [torch.zeros((seq_len, head, head_dim)) for _ in range(layer)]

print("\n---------------- 故事开始 ----------------")
current_token_id = 1
max_step = 100
start_time = time.time()

# 🚀 使用上下文管理器，这是 PyTorch 最稳的推理模式写法
with torch.no_grad():
    for pos in range(max_step):
        step_start = time.time()

        x = embed_w[current_token_id]

        for i in range(layer):
            # Attention
            x_norm = rmsnorm(x, attn_norm_w[i])
            q = F.linear(x_norm, wq_w[i]).view(head, head_dim)
            k = F.linear(x_norm, wk_w[i]).view(head, head_dim)
            v = F.linear(x_norm, wv_w[i]).view(head, head_dim)

            d_tensor = torch.arange(0, head_dim, 2, dtype=torch.float32)
            freqs = 1.0 / (10000.0 ** (d_tensor / head_dim))
            angles = pos * freqs
            cos_val = torch.cos(angles).unsqueeze(0)
            sin_val = torch.sin(angles).unsqueeze(0)

            q_view = q.view(head, head_dim // 2, 2)
            k_view = k.view(head, head_dim // 2, 2)

            q_rot = torch.empty_like(q_view)
            q_rot[:, :, 0] = q_view[:, :, 0] * cos_val - q_view[:, :, 1] * sin_val
            q_rot[:, :, 1] = q_view[:, :, 0] * sin_val + q_view[:, :, 1] * cos_val
            q = q_rot.view(head, head_dim)

            k_rot = torch.empty_like(k_view)
            k_rot[:, :, 0] = k_view[:, :, 0] * cos_val - k_view[:, :, 1] * sin_val
            k_rot[:, :, 1] = k_view[:, :, 0] * sin_val + k_view[:, :, 1] * cos_val
            k = k_rot.view(head, head_dim)

            k_cache[i][pos] = k
            v_cache[i][pos] = v

            k_past = k_cache[i][:pos+1].transpose(0, 1)
            v_past = v_cache[i][:pos+1].transpose(0, 1)

            scores = torch.bmm(q.unsqueeze(1), k_past.transpose(1, 2)) / math.sqrt(head_dim)
            probs = F.softmax(scores, dim=-1)
            out = torch.bmm(probs, v_past).view(-1)

            x = x + F.linear(out, wo_w[i])

            # MLP
            x_norm = rmsnorm(x, ffn_norm_w[i])
            gate = F.linear(x_norm, w1_w[i])
            up = F.linear(x_norm, w3_w[i])
            hidden = F.silu(gate) * up
            x = x + F.linear(hidden, w2_w[i])

        x = rmsnorm(x, final_norm_w)
        logits = F.linear(x, lm_head_w)

        # 找词
        next_token_id = torch.argmax(logits).item()
        word = vocab[next_token_id]

        step_us = (time.time() - step_start) * 1_000_000
        tk_s = 1_000_000.0 / step_us if step_us > 0 else 0

        if word == '\n':
            print(word, end='', flush=True)
        else:
            print(f"{word}\033[90m[{tk_s:.2f} tk/s]\033[0m", end='\n', flush=True)

        current_token_id = next_token_id

duration = time.time() - start_time
print("\n\n---------------- 故事结束 ----------------")
print(f"⏱️  总耗时: {duration:.3f} 秒")
print(f"🚀 PyTorch 平均生成速度: {max_step / duration:.2f} tokens/秒")