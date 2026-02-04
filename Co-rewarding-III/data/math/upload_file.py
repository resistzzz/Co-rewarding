#!/usr/bin/env python3
"""
示例：将两个 parquet 里同名的一一对应列横向拼接成 pair 数据集。
你可以把 COLS 列表改成你自己的列名即可。
输出：data/math/train_pairs.parquet
"""

import pandas as pd
from pathlib import Path

# === 配置区 ===
ORIGINAL_FILE = "coreward/data/math/train_original.parquet"
REWRITE_FILE  = "coreward/data/math/train_rewrite_Qwen3-32B.parquet"
OUTPUT_FILE   = "coreward/data/math/train_pairs.parquet"


def make_pairs() -> None:
    """横向拼接指定的列。"""
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    df_orig = pd.read_parquet(ORIGINAL_FILE)
    df_rw   = pd.read_parquet(REWRITE_FILE)

    if len(df_orig) != len(df_rw):
        raise ValueError("两个文件行数不一致，无法一一配对！")

    new_rows = []
    for idx, (orig_row, rw_row) in enumerate(zip(df_orig.itertuples(index=False), df_rw.itertuples(index=False))):
        # ========= 这里开始你来写 =========
        # orig_row / rw_row 都是 namedtuple，可直接 .列名 访问
        # 示例：把两列拼成 JSON 字符串
        assert orig_row.reward_model['ground_truth'] == rw_row.reward_model['ground_truth']
        new_row = {
            "idx": idx,
            "original_question": orig_row.prompt[-1]['content'],
            "rephrased_question": rw_row.prompt[-1]['content'],
            "solution": orig_row.reward_model['ground_truth'],
            "level": orig_row.level,
            "type": orig_row.type,
            "original_message": orig_row.prompt,
            "rephrased_message": rw_row.prompt
        }
        # ==================================

        new_rows.append(new_row)

    # 4. 转成 DataFrame 并保存
    new_df = pd.DataFrame(new_rows)
    out_path = Path(OUTPUT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_parquet(out_path, index=False)


if __name__ == "__main__":
    make_pairs()