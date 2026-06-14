#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
内存优化版：合并三个模块的 Lightning checkpoint，避免 OOM。
用法: python scripts/merge_ckpts_lowmem.py \
          --sr_ckpt sr-checkpoints/last.ckpt \
          --diff_ckpt diff-checkpoints/last.ckpt \
          --enhance_ckpt enhance-checkpoints/last.ckpt \
          --output checkpoints/merged_all.ckpt \
          --base_ckpt diff-checkpoints/last.ckpt
"""

import argparse
import gc
import torch


def load_state_dict_only(path):
    """只加载 checkpoint 的 state_dict，不保留其他元数据，节省内存"""
    print(f"  加载 {path} ...")
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["state_dict"]
    # 立刻删除 ckpt 释放其他部分（如 optimizer_states）的内存
    del ckpt
    gc.collect()
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr_ckpt", type=str, default=None)
    parser.add_argument("--diff_ckpt", type=str, default=None)
    parser.add_argument("--enhance_ckpt", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--base_ckpt", type=str, default=None,
                        help="基础 checkpoint（提供元数据及其他参数，默认使用 diff_ckpt）")
    args = parser.parse_args()

    # 确定 base_ckpt 路径
    base_path = args.base_ckpt
    if base_path is None:
        for p in [args.diff_ckpt, args.sr_ckpt, args.enhance_ckpt]:
            if p is not None:
                base_path = p
                break
    if base_path is None:
        raise ValueError("至少需要一个 checkpoint 作为基础")

    # 1. 加载 base_ckpt 的 state_dict 作为起点（这是合并的基础）
    print(f"加载基础 checkpoint: {base_path}")
    base_ckpt = torch.load(base_path, map_location="cpu")
    base_state = base_ckpt["state_dict"]
    # 我们之后会替换 merged_state，所以这里先复制一份给 merged_state
    merged_state = base_state.copy()
    # 删除 base_ckpt 的其他数据以释放内存，但保留一份元数据用于最终保存
    base_meta = {k: v for k, v in base_ckpt.items() if k != "state_dict"}
    del base_state, base_ckpt
    gc.collect()

    # 2. 定义各模块的路径与对应的前缀
    module_info = [
        ("model_sr", args.sr_ckpt),
        ("model_diff", args.diff_ckpt),
        ("model_enhance", args.enhance_ckpt),
    ]

    for prefix, ckpt_path in module_info:
        if ckpt_path is None:
            print(f"  模块 {prefix} 未提供 checkpoint，保留基础权重")
            continue
        print(f"合并 {prefix} 从 {ckpt_path} ...")
        # 仅加载 state_dict
        new_state = load_state_dict_only(ckpt_path)
        # 遍历该模块的 key 并更新
        updated = 0
        for key in list(merged_state.keys()):
            if key.startswith(f"{prefix}."):
                if key in new_state:
                    merged_state[key] = new_state[key]
                    updated += 1
                # 如果 new_state 中没有，则保留基础权重（默认为原来的值）
        # 删除新加载的 state 以释放内存
        del new_state
        gc.collect()
        print(f"  已更新 {updated} 个键")

    # 3. 构建最终的 checkpoint，清除优化器状态
    final_ckpt = base_meta.copy()
    final_ckpt["state_dict"] = merged_state
    # 移除可能残留的优化器/调度器状态
    final_ckpt.pop("optimizer_states", None)
    final_ckpt.pop("lr_schedulers", None)
    # 清理临时变量
    del merged_state, base_meta
    gc.collect()

    # 4. 保存（仅写入一次，不会额外占用太多内存）
    print(f"保存合并后 checkpoint 到 {args.output} ...")
    torch.save(final_ckpt, args.output)
    print("✅ 完成！")


if __name__ == "__main__":
    main()