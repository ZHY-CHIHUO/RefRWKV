import argparse
import torch

def main():
    parser = argparse.ArgumentParser(
        description="合并 SR-only 和 Diff-only 的 Lightning checkpoint，生成包含两者权重的完整 checkpoint"
    )
    parser.add_argument("--sr_ckpt", type=str, required=True,
                        help="只训练了 model_sr 的 checkpoint 路径")
    parser.add_argument("--diff_ckpt", type=str, required=True,
                        help="只训练了 model_diff 的 checkpoint 路径")
    parser.add_argument("--output", type=str, required=True,
                        help="输出的合并后 checkpoint 路径")
    parser.add_argument("--base_ckpt", type=str, default=None,
                        help="用作元数据基础的 checkpoint ")
    args = parser.parse_args()

    # 1. 加载所有 checkpoint
    sr_ckpt = torch.load(args.sr_ckpt, map_location="cpu")
    diff_ckpt = torch.load(args.diff_ckpt, map_location="cpu")
    base_ckpt = diff_ckpt if args.base_ckpt is None else torch.load(args.base_ckpt, map_location="cpu")

    sr_state = sr_ckpt["state_dict"]
    diff_state = diff_ckpt["state_dict"]
    base_state = base_ckpt["state_dict"]

    # 2. 合并 state_dict
    merged_state = {}
    missing_sr = 0
    missing_diff = 0

    for key, base_val in base_state.items():
        if key.startswith("model_sr."):
            if key in sr_state:
                merged_state[key] = sr_state[key]
            else:
                print(f"[警告] 在 sr_ckpt 中找不到 {key}，使用 base_ckpt 中的值")
                merged_state[key] = base_val
                missing_sr += 1
        elif key.startswith("model_diff."):
            if key in diff_state:
                merged_state[key] = diff_state[key]
            else:
                print(f"[警告] 在 diff_ckpt 中找不到 {key}，使用 base_ckpt 中的值")
                merged_state[key] = base_val
                missing_diff += 1
        else:
            # 其他参数（model_enhance、损失权重等）直接保留 base_ckpt 的值
            merged_state[key] = base_val

    print(f"合并完成。model_sr 权重从 sr_ckpt 复制，缺失 {missing_sr} 个键；"
          f"model_diff 权重从 diff_ckpt 复制，缺失 {missing_diff} 个键。")

    # 3. 更新 base_ckpt 的 state_dict
    base_ckpt["state_dict"] = merged_state

    # 4. 清除优化器状态（因为合并后通常要开始新的训练阶段）
    base_ckpt.pop("optimizer_states", None)
    base_ckpt.pop("lr_schedulers", None)

    # 5. 保存
    torch.save(base_ckpt, args.output)
    print(f"✅ 合并后的 checkpoint 已保存至: {args.output}")

if __name__ == "__main__":
    '''

    python scripts/merge_sr_diff_ckpt.py \
    --sr_ckpt checkpoints/sr_only.ckpt \
    --diff_ckpt checkpoints/diff_only.ckpt \
    --output checkpoints/merged.ckpt

    '''
    main()

