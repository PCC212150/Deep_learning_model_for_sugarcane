"""单张图片预测的共享实现（test.py / inference.py / tool/tune_stats 复用）。

这里同时是**统计口径的唯一实现**：「根系掩码只在检查范围内统计」这件事只在这里算一次，
避免出现「调参工具扫出来的最优参数 ≠ 部署时实际用的口径」。

流程：
1. 缩放到模型输入尺寸 -> 前向 -> 逐通道 sigmoid 概率（多标签，三类不互斥）；
2. 检查范围通道二值化 -> 取最大连通域 -> 按行/列覆盖量剖面拟合成矩形（ROI，
   见 _fit_rect：直接取外接矩形会被托盘外的碎片拉大），再按 CHECK_MARGIN_PX 向外膨胀
   （防止把贴着框边的根切断，切一根会被数成两根）；
3. **在概率层**把 ROI 之外的根系概率清零，再做滞回阈值（顺序很重要：先清零再阈值，
   否则框外的弱响应会把框内两段连通起来）；
4. 结果再与 ROI 精确求交（上采样会让边界有一圈渐变带）。
"""
import numpy as np
import torch
from skimage.measure import label as _label

import config
from common import image_io

# 通道序号（与 config.CLASS_NAMES 一致）
CH_ROOT, CH_STEM, CH_CHECK = 0, 1, 2


def _largest_cc(binary: np.ndarray):
    """最大连通域掩码；全空返回 None。"""
    if binary is None or not binary.any():
        return None
    lab = _label(binary, connectivity=2)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0                      # 背景不算
    if sizes.size < 2 or sizes.max() == 0:
        return None
    return lab == int(sizes.argmax())


def _largest_bbox(binary: np.ndarray):
    """二值掩码最大连通域的外接矩形 (x0, y0, x1, y1)，右/下为开区间；空则 None。"""
    cc = _largest_cc(binary)
    if cc is None:
        return None
    ys, xs = np.nonzero(cc)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _fit_rect(binary: np.ndarray, rel: float = 0.5, q: float = 95.0):
    """把「检查范围」预测拟合成一个矩形，返回模型分辨率下的 (x0, y0, x1, y1)。

    不能直接取最大连通域的外接矩形：预测掩码在托盘之外常有连通的边缘/玻璃/水面碎片
    （面积占比只差几个百分点，但会把外接矩形拉到接近整图）。实测在 14 张有标注的图上，
    外接矩形与标注框的平均 IoU 只有 0.84（最差 0.64），本函数是 0.98。

    做法：先取最大连通域，再看**行/列方向的覆盖量剖面** —— 托盘内部每行几乎铺满，
    框外的碎片行只覆盖一小段。以剖面的 95 分位数为基准，只保留覆盖量 ≥ rel 倍的行/列，
    取这些行/列构成的矩形。对 rel（0.4~0.6）与 q（90~99）都不敏感（IoU 波动 <0.004）。
    """
    cc = _largest_cc(binary)
    if cc is None:
        return None
    rows = cc.sum(axis=1).astype(np.float64)      # 每行的前景像素数
    cols = cc.sum(axis=0).astype(np.float64)
    rows_nz, cols_nz = rows[rows > 0], cols[cols > 0]
    if rows_nz.size == 0 or cols_nz.size == 0:
        return _largest_bbox(binary)
    keep_r = np.nonzero(rows >= rel * np.percentile(rows_nz, q))[0]
    keep_c = np.nonzero(cols >= rel * np.percentile(cols_nz, q))[0]
    if keep_r.size == 0 or keep_c.size == 0:
        return _largest_bbox(binary)
    return (int(keep_c.min()), int(keep_r.min()),
            int(keep_c.max()) + 1, int(keep_r.max()) + 1)


def _bbox_to_orig(box, w1: int, h1: int, w0: int, h0: int, margin_px: float):
    """模型分辨率的外接矩形 -> 原图坐标，向外取整并按 margin_px 膨胀。"""
    sx, sy = w0 / float(w1), h0 / float(h1)
    x0 = max(0, int(np.floor(box[0] * sx - margin_px)))
    y0 = max(0, int(np.floor(box[1] * sy - margin_px)))
    x1 = min(w0, int(np.ceil(box[2] * sx + margin_px)))
    y1 = min(h0, int(np.ceil(box[3] * sy + margin_px)))
    return x0, y0, x1, y1


def _bbox_to_model(box, w1: int, h1: int, w0: int, h0: int):
    """原图坐标的外接矩形 -> 模型分辨率（同样向外取整，保证不内缩）。"""
    sx, sy = w1 / float(w0), h1 / float(h0)
    x0 = max(0, int(np.floor(box[0] * sx)))
    y0 = max(0, int(np.floor(box[1] * sy)))
    x1 = min(w1, int(np.ceil(box[2] * sx)))
    y1 = min(h1, int(np.ceil(box[3] * sy)))
    return x0, y0, x1, y1


def _forward_prob(model, x):
    """前向 + sigmoid，返回概率张量 [1,C,h1,w1]。

    model 可以是单个模型，也可以是**模型列表**（集成）：逐个前向，对概率取平均。
    集成必须在同一个输入网格上做，所以调用方要保证各模型的训练 --size 一致
    （ckpt.load_models 会校验）。多个模型断的地方不一样，平均后碎片更容易连上。
    """
    models = list(model) if isinstance(model, (list, tuple)) else [model]
    acc = None
    for m in models:
        m.eval()
        with torch.no_grad():
            p = torch.sigmoid(m(x))
        acc = p if acc is None else acc + p
    return acc / len(models)          # 单个模型时除以 1，与原行为完全一致


def predict(model, img: np.ndarray, max_side: int, stride: int = 16,
            device="cuda", low_thresh: float = 0.10,
            check_margin_px: float = None, use_check: bool = True,
            full_channels=(CH_ROOT, CH_STEM)) -> dict:
    """对一张 uint8 RGB (h0, w0, 3) 图片做多通道分割预测。

    model 可以是单个模型或模型列表（列表 = 集成，概率平均，见 _forward_prob）。

    low_thresh > 0 时根系用滞回阈值（细弱处断段接回，适合根数/长度统计），否则用 0.5。
    use_check=False 时不做检查范围限定（用于没有该标注/对比旧口径）。
    full_channels 指定 masks 里哪些通道要算**全分辨率**二值掩码（默认根系+茎，
    这两条是全项目唯二有人读的）；不在其中的通道在 masks 里是 None。**注意
    probs 不受影响**，永远是全通道的模型分辨率概率 —— 像素指标用的是它。

    返回：
        probs        [C 个 float32 ndarray (h1,w1)]，模型分辨率的逐通道概率（**原始**，
                     未做 ROI 处理）——像素指标用它与 GT 比较，与训练时的验证口径一致
        masks        [C 个 bool ndarray (h0,w0)]，上采样回原图并二值化；第 0 个 =
                     mask_counted（根系，已限定 ROI），其余为各通道原始阈值结果
        mask_counted bool (h0,w0)  最终用于统计的根系掩码（= 根系 ∩ ROI）
        check_box    (x0,y0,x1,y1) 原图坐标的 ROI（含 margin）；未启用/失败为 None
        check_ok     bool          检查范围是否可用（预测失败会退回全图并置 False）
        prob_target  根系通道概率（float32 ndarray (h1,w1)，**已在概率层把 ROI 之外清零**）；
                     供调参工具重新阈值化时复用同一口径
        mask_orig    兼容旧调用：= mask_counted
        target_size  (w1, h1)
    """
    if check_margin_px is None:
        check_margin_px = config.CHECK_MARGIN_PX
    h0, w0 = img.shape[:2]
    w1, h1 = image_io.target_size(w0, h0, max_side, stride)
    small = image_io.resize_rgb(img, w1, h1)
    x = image_io.to_model_input(small).to(device)
    prob = _forward_prob(model, x)
    n_ch = prob.shape[1]
    probs = [prob[0, c].float().cpu().numpy() for c in range(n_ch)]

    # ---- 检查范围：最大连通域 -> 外接矩形 -> 面积合理性校验 ----
    check_box, check_ok = None, False
    prob_root = probs[CH_ROOT]
    if use_check and n_ch > CH_CHECK:
        box_m = _fit_rect(probs[CH_CHECK] > 0.5)
        if box_m is not None:
            box_o = _bbox_to_orig(box_m, w1, h1, w0, h0, check_margin_px)
            ratio = ((box_o[2] - box_o[0]) * (box_o[3] - box_o[1])) / float(w0 * h0)
            if config.CHECK_MIN_RATIO <= ratio <= config.CHECK_MAX_RATIO:
                check_box, check_ok = box_o, True
            else:
                print(f"[警告] 检查范围预测异常（占图面 {ratio:.1%}），本图退回全图统计")
        else:
            print("[警告] 检查范围通道没有预测出任何前景，本图退回全图统计")

    # ---- 根系：概率层先清 ROI 之外，再做阈值 ----
    root_p = prob_root.copy()
    if check_ok:
        x0m, y0m, x1m, y1m = _bbox_to_model(check_box, w1, h1, w0, h0)
        keep = np.zeros_like(root_p, dtype=bool)
        keep[y0m:y1m, x0m:x1m] = True
        root_p[~keep] = 0.0
    prob_root_t = torch.from_numpy(root_p).unsqueeze(0).unsqueeze(0)  # [1,1,h1,w1]
    if low_thresh and low_thresh > 0:
        mask_counted = image_io.prob_to_orig_mask_hysteresis(
            prob_root_t, w0, h0, high=0.5, low=low_thresh, channel=0)
    else:
        mask_counted = image_io.prob_to_orig_mask(prob_root_t, w0, h0,
                                                 threshold=0.5, channel=0)
    root_ok = True
    if check_ok:                       # 精确求交（上采样后边界有一圈渐变带）
        exact = np.zeros((h0, w0), dtype=bool)
        exact[check_box[1]:check_box[3], check_box[0]:check_box[2]] = True
        mask_counted &= exact
        # ---- 兜底：滞回低阈值是否把整片检查区淹了 ----
        # 低阈值的前提是「背景概率接近 0」，但这个前提没人校验过。实测某模型背景概率
        # 中位数涨到 0.125（低阈值是 0.10），检查框内 86% 被判成根、整张图糊成一片。
        # 健康模型的这个比例只有 0.96%~5.50%（7 张测试图实测），所以超阈值一定是异常，
        # 退回只用高阈值 0.5 重算。宁可少统计，也不要给出一片假根。
        roi_area = float((check_box[2] - check_box[0]) * (check_box[3] - check_box[1]))
        if roi_area > 0:
            cov = float(mask_counted[check_box[1]:check_box[3],
                                     check_box[0]:check_box[2]].sum()) / roi_area
            if cov > config.PRED_MAX_ROOT_RATIO:
                print(f"[警告] 滞回低阈值({low_thresh})把检查区淹没了：根占检查框 "
                      f"{cov:.1%}（上限 {config.PRED_MAX_ROOT_RATIO:.0%}），"
                      f"退回只用高阈值 0.5 重算。通常是模型的背景概率有底噪"
                      f"（训练不充分或域不匹配），换个模型比调阈值管用。")
                mask_counted = image_io.prob_to_orig_mask(
                    prob_root_t, w0, h0, threshold=0.5, channel=0)
                mask_counted &= exact
                root_ok = False

    # ---- 其余通道：普通 0.5 阈值（茎/检查范围是块状目标，不需要滞回） ----
    # 只算**真的有人用**的通道：全分辨率二值化一张要 ~100ms。逐处查过调用方 ——
    # inference / test / tune_stats 都只用 CH_STEM，check 用的是模型分辨率上算出来的
    # check_box，那个全分辨率掩码全项目没人读。传 None 占位，保住 masks 的下标语义。
    masks = [None] * n_ch
    masks[CH_ROOT] = mask_counted
    for c in full_channels:
        if c != CH_ROOT and c < n_ch:
            masks[c] = image_io.prob_to_orig_mask(prob, w0, h0,
                                                  threshold=0.5, channel=c)

    out = {
        "probs": probs,
        "masks": masks,
        "mask_counted": mask_counted,
        "check_box": check_box,
        "check_ok": check_ok,
        "root_ok": root_ok,        # False = 滞回被背景底噪淹没，已退回高阈值 0.5
        "target_size": (w1, h1),
        # ---- 兼容旧调用 ----
        "prob_target": root_p,                      # (h1, w1) float32，已清 ROI 之外
        "mask_orig": mask_counted,
    }
    return out
