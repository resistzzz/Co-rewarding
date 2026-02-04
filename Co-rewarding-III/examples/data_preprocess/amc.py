#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将形如 [{"prompt": "...", "answer": "...", "source": "...", "id": "..."}] 的 JSON/JSONL
转换为参考代码所需的 parquet（含 prompt 消息格式、reward_model 等）。
用法示例：
  python convert_openrs_like_json_to_parquet.py \
      --input /mnt/data/test\ (1).json \
      --local_dir data/open-rs \
      --data_source knoveleng/open-rs \
      --ability math
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any, Iterable

import datasets


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    既支持：
      1) 一个 JSON 数组文件
      2) JSON Lines（每行一个 JSON 对象）
    """
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2048)
        f.seek(0)
        head_stripped = head.lstrip()

        # 粗略判断是否为数组 JSON
        if head_stripped.startswith("["):
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("顶层 JSON 不是数组。")
            return data

        # 否则按 JSONL 读
        items = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
        return items


def build_row(
    ex: Dict[str, Any],
    idx: int,
    instruction_following: str,
    data_source: str,
    ability: str,
) -> Dict[str, Any]:
    """
    将一条 {prompt, answer, source, id} -> parquet 目标格式
    目标字段：
      - data_source: str
      - prompt: [{"role":"system","content":...}, {"role":"user","content":...}]
      - ability: e.g. "math"
      - reward_model: {"style":"rule", "ground_truth": <answer>}
      - extra_info: {"split":..., "index":..., "source":..., "id":...}
    """
    question = ex.get("prompt", "")
    answer = ex.get("answer", "")
    source = ex.get("source", None)
    ex_id = ex.get("id", None)

    # 与参考代码保持一致的 system 提示
    # Let's think step by step and output the final answer within \boxed{}.
    sys_prompt = instruction_following

    row = {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question},
        ],
        "ability": ability,
        "reward_model": {"style": "rule", "ground_truth": str(answer)},
        "extra_info": {
            "index": idx,
            "source": source,
            "id": ex_id,
        },
    }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../Spurious_Rewards/code/data/AMC-TTT/test.json", help="输入 JSON/JSONL 文件路径")
    parser.add_argument("--local_dir", default="data/AMC-TTT", help="本地输出目录")
    parser.add_argument("--hdfs_dir", default=None, help="（可选）HDFS 目的目录")
    parser.add_argument(
        "--data_source",
        default="DigitalLearningGmbH/MATH-lighteval",
        help="写入 data_source 字段的来源名（可自定义）",
    )
    parser.add_argument(
        "--ability",
        default="math",
        help="写入 ability 字段（默认 math）",
    )
    parser.add_argument(
        "--system_prompt",
        default="Let's think step by step and output the final answer within \\boxed{}.",
        help="system 提示（默认与参考代码一致）",
    )
    args = parser.parse_args()

    # 读取数据
    print(f"Loading input from {args.input} ...", flush=True)
    items = read_json_or_jsonl(args.input)
    if not items:
        print("输入为空，未生成任何数据。", file=sys.stderr)
        sys.exit(1)

    # 构造目标 rows
    rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(items):
        rows.append(
            build_row(
                ex=ex,
                idx=i,
                instruction_following=args.system_prompt,
                data_source=args.data_source,
                ability=args.ability,
            )
        )

    # 转成 HF Dataset
    dataset = datasets.Dataset.from_list(rows)

    # 写 parquet
    os.makedirs(args.local_dir, exist_ok=True)
    out_path = os.path.join(args.local_dir, f"test.parquet")
    print(f"Writing parquet to {out_path} ...", flush=True)
    dataset.to_parquet(out_path)

    # 可选：同步到 HDFS
    if args.hdfs_dir:
        try:
            from verl.utils.hdfs_io import copy, makedirs  # type: ignore
            print(f"Copying to HDFS: {args.hdfs_dir} ...", flush=True)
            makedirs(args.hdfs_dir)
            copy(src=args.local_dir, dst=args.hdfs_dir)
        except Exception as e:
            print(
                f"[WARN] 复制到 HDFS 失败（可能本地没有 verl.hdfs_io 或 HDFS 环境未配置）：{e}",
                file=sys.stderr,
            )

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
