"""二值分割指标：IoU / Dice / 像素准确率（在预测掩码与 GT 掩码间计算）。"""
import numpy as np


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """pred/gt: 同形状 bool 掩码。返回 {'iou','dice','accuracy'}。"""
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    tn = float(pred.size - tp - fp - fn)
    denom = tp + fp + fn
    iou = tp / denom if denom > 0 else 0.0
    dice = 2.0 * tp / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
    acc = (tp + tn) / float(pred.size) if pred.size else 0.0
    return {"iou": iou, "dice": dice, "accuracy": acc}
