import argparse
import numpy as np
from sewar.full_ref import (psnr, ssim, sam, rmse)

# ==============================================================================
# 评估核心函数
# ==============================================================================
def evaluate(pred: np.ndarray, 
             gt: np.ndarray, 
             max_val: float = 1.0) -> dict:
    """
    计算全参考图像质量指标。

    参数:
        pred : 预测图像， (H, W, C) 或 (H, W)，范围 [0, max_val]
        gt   : 参考图像， (H, W, C) 或 (H, W)，范围 [0, max_val]
        max_val: 数据的最大像素值，通常 1.0 或 255.0

    返回:
        字典，包含 PSNR, SSIM, SAM, ERGAS, CC, RMSE, UQI 等 (缺失波段时可能跳过)
    """
    # 确保数据类型和维度一致
    assert pred.shape == gt.shape, f"形状不一致: pred {pred.shape}, gt {gt.shape}"
    pred = pred.astype(np.float64)
    gt   = gt.astype(np.float64)

    results = {}

    # ---- PSNR ----
    results['PSNR'] = psnr(gt, pred, MAX=max_val)

    # ---- SSIM (多通道均值) ----
    # sewar 的 ssim 直接支持多通道图像 (H,W,C)
    results['SSIM'] = ssim(gt, pred, MAX=max_val)[0]

    # ---- SAM (光谱角，单位: 度) ----
    # 仅当图像有多个波段时计算，否则设为 None
    if pred.ndim == 3 and pred.shape[2] > 1:
        results['SAM'] = sam(gt, pred)
    else:
        results['SAM'] = None

    # ---- RMSE ----
    results['RMSE'] = rmse(gt, pred)

    return results


def print_metrics(metrics: dict, title: str = "Evaluation Results"):
    """美观地打印指标字典。"""
    print(f"\n{'='*40}")
    print(f"  {title}")
    print('='*40)
    for k, v in metrics.items():
        if v is not None:
            # 根据值的大小选择合适的小数位数
            if isinstance(v, (int, float)):
                if abs(v) < 0.01:
                    print(f"  {k:8s} : {v:.6f}")
                else:
                    print(f"  {k:8s} : {v:.4f}")
            else:
                print(f"  {k:8s} : {v}")
        else:
            print(f"  {k:8s} : N/A (计算失败或不适用)")
    print('='*40)

def evaluate_CHW(pred: np.ndarray,
                 gt: np.ndarray,
                 max_val: float = 1.0,
                 print_result: bool = True,
                 title: str = "Evaluation (CHW -> HWC)") -> dict:
    """
    接收 (C, H, W) 格式的图像，自动转换为 (H, W, C) 后评估并打印结果。
    这是为了方便深度学习 pipeline 中直接使用模型输出（通常为 CHW 格式）。
    
    参数:
        pred, gt : 形状为 (C, H, W) 或 (H, W) 的单/多波段图像
        print_result : 是否打印结果
        title : 打印时的标题
    返回:
        指标字典
    """
    # 维度转换：如果输入是 3D 且第一维为通道数，则转置
    if pred.ndim == 3:
        # 假设输入是 (C, H, W)，转为 (H, W, C)
        pred = np.transpose(pred, (1, 2, 0))
        gt   = np.transpose(gt, (1, 2, 0))
    # 如果是 2D 单波段，直接使用
    
    metrics = evaluate(pred, gt, max_val=max_val)
    
    if print_result:
        print_metrics(metrics, title=title)
    
    return metrics

def average_metrics(results_list: list[dict]) -> dict:
    """
    对每个指标求平均，自动跳过 None 值。
    """
    avg_res = {}
    if not results_list:
        return avg_res

    for k in results_list[0].keys():
        # 过滤掉 None
        valid_values = [r[k] for r in results_list if r[k] is not None]
        if len(valid_values) == 0:
            avg_res[k] = None
        else:
            avg_res[k] = float(np.mean(valid_values))
    return avg_res


# ==============================================================================
# 示例：直接运行或作为模块使用
# ==============================================================================
if __name__ == '__main__':
    pred = np.random.rand(10,2,2)
    gt   = np.random.rand(10,2,2)
    print(gt)
    print(pred)
    res = evaluate_CHW(pred, gt)
    print(res)