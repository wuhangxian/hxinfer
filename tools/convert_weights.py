"""
convert_weights.py

把 HuggingFace 的 pytorch_model.bin（pickle 格式）转换成
hxinfer C++ 引擎能直接 mmap 读取的格式：

  输出1: llama7b_weights.bin   —— 所有权重的 float16 原始字节，顺序拼接
  输出2: llama7b_index.json    —— 每个权重的名字、shape、在 bin 文件里的偏移量

用法:
  python tools/convert_weights.py \
    --model_dir /workspace/models/Yarn-Llama-2-7b-128k \
    --output_dir /workspace/whx
"""

import argparse
import json
import os
import torch


def convert(model_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    index_file = os.path.join(model_dir, "pytorch_model.bin.index.json")
    with open(index_file) as f:
        hf_index = json.load(f)

    # 每个 shard 文件名 -> 该 shard 包含哪些权重名
    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in hf_index["weight_map"].items():
        shard_to_keys.setdefault(shard, []).append(key)

    output_bin  = os.path.join(output_dir, "llama7b_weights.bin")
    output_json = os.path.join(output_dir, "llama7b_index.json")

    index = {}   # 最终写入 llama7b_index.json 的内容
    offset = 0   # 当前写到 bin 文件的第几个字节

    print(f"输出路径: {output_bin}")
    print(f"共 {len(hf_index['weight_map'])} 个权重张量，开始转换...\n")

    with open(output_bin, "wb") as fout:
        for shard_name in sorted(shard_to_keys.keys()):
            shard_path = os.path.join(model_dir, shard_name)
            print(f"  加载 {shard_name} ...")

            shard = torch.load(shard_path, map_location="cpu", weights_only=True)

            for key in shard_to_keys[shard_name]:
                tensor = shard[key]

                # bfloat16 / float32 → float16
                tensor = tensor.to(torch.float16).contiguous()

                raw_bytes = tensor.numpy().tobytes()
                nbytes    = len(raw_bytes)

                fout.write(raw_bytes)

                index[key] = {
                    "offset": offset,
                    "nbytes": nbytes,
                    "shape":  list(tensor.shape),
                    "dtype":  "float16",
                }

                offset += nbytes

            del shard  # 释放这个 shard 的内存

    with open(output_json, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\n完成！")
    print(f"  权重文件: {output_bin}  ({offset / 1e9:.1f} GB)")
    print(f"  索引文件: {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",  required=True,  help="HuggingFace 模型目录")
    parser.add_argument("--output_dir", required=True,  help="输出目录")
    args = parser.parse_args()
    convert(args.model_dir, args.output_dir)
