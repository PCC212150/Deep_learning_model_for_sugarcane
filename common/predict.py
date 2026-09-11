"""单张图片预测的共享实现（test.py 与 infrernce.py 复用）。"""
import numpy as np
import torch

from common import image_io


def predict(model, img: np.ndarray, max_side: int, stride: int = 16,
            device="cuda", low_thresh: float = 0.15) -> dict:
    """对一张 uint8 RGB (h0, w0, 3) 图片做分割预测。

    low_thresh > 0 时用滞回阈值生成 mask_orig（细弱处断段接回，
    适合根数/长度统计）；置 0 则用普通 0.5 阈值。
    返回 {"prob_target": np.float32 (h1, w1),   # 模型分辨率概率图
          "mask_orig":   bool  (h0, w0),        # 上采样回原图并二值化
          "target_size": (w1, h1)}
    """
    h0, w0 = img.shape[:2]
    w1, h1 = image_io.target_size(w0, h0, max_side, stride)
    small = image_io.resize_rgb(img, w1, h1)
    x = image_io.to_model_input(small).to(device)
    model.eval()
    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit)
    prob_t = prob[0, 0].float().cpu().numpy()          # (h1, w1)
    if low_thresh and low_thresh > 0:
        mask_orig = image_io.prob_to_orig_mask_hysteresis(
            prob, w0, h0, high=0.5, low=low_thresh)
    else:
        mask_orig = image_io.prob_to_orig_mask(prob, w0, h0, threshold=0.5)
    return {"prob_target": prob_t, "mask_orig": mask_orig, "target_size": (w1, h1)}
