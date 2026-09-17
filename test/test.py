"""在测试集（datasets/root/test，每组含 图片 + rsml + labelme json 真值）上评测已训练的模型。

用法（readme 两种写法均支持）：
    python test/test.py                                   # 自动使用最新模型
    python test/test.py --model model_202609091135
    python test/test.py --model_202609091135              # 兼容写法

输出：三类通道（根系 / 茎横截面 / 检查范围）各自的 IoU / Dice / 像素准确率，
以及每张图「预测根数/总长 vs RSML 真值」的对比与平均绝对误差。
结果保存至模型文件夹内 model_test_{年月日时分}.csv（UTF-8 BOM，Excel 可直接打开；
文件末尾是若干以 # 开头的汇总行）。
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from common import ckpt, image_io, metrics, naming, predict  # noqa: E402
from common.dataset import (CH_CHECK, CH_ROOT, CH_STEM, build_target_masks,  # noqa: E402
                            discover_pairs, find_other)
from common.rsml_parse import parse_rsml, root_stats  # noqa: E402
from common.skeleton_stats import analyze_mask_counted  # noqa: E402


def preprocess_argv():
    """把 --model_xxx 兼容为 --model model_xxx。"""
    out, i = [], 0
    argv = sys.argv[1:]
    while i < len(argv):
        t = argv[i]
        if t.startswith("--model_") and "=" not in t:
            out += ["--model", t[2:]]
        else:
            out.append(t)
        i += 1
    return out


def parse_args():
    p = argparse.ArgumentParser(description="测试甘蔗根系 U-Net（三通道）")
    p.add_argument("--model", default=None,
                   help="模型文件夹名(可省略 model_ 前缀)；缺省自动取最新")
    p.add_argument("--size", type=int, default=None,
                   help="模型输入长边像素；缺省用模型训练时的 --size（自动从权重读），"
                        "再缺省 config.MAX_SIDE")
    p.add_argument("--data-dir", type=Path, default=config.TEST_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=config.MODEL_DIR)
    p.add_argument("--mm-per-px", type=float, default=None,
                   help="长度换算：1 像素 = 多少毫米（默认用 config.MM_PER_PX）")
    p.add_argument("--no-anchor", dest="anchor", action="store_false",
                   default=config.STEM_ANCHOR,
                   help="统计不把折线起点补到茎（标注只画看得见的根时用）。"
                        "只影响总长，不影响 Dice/根数；缺省见 config.STEM_ANCHOR")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args(preprocess_argv())


def main():
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available()
                          else "cuda")
    # --model 支持逗号分隔的多个模型（集成），见 ckpt.resolve_pths
    pths, names = ckpt.resolve_pths(args.model, args.out_dir)
    pth = pths[0]
    print(f"模型目录: {pth.parent}")
    if len(names) > 1:
        print(f"集成 {len(names)} 个: {' + '.join(names)}")
    model, metas = ckpt.load_models(pths, device)     # 集成时 model 是模型列表
    meta = metas[0]
    print(f"权重: {pth.name} (保存于 epoch {meta.get('epoch', '?')}"
          + (f"，训练验证Dice {meta['val_dice']:.4f}" if meta.get("val_dice") else "")
          + f"，输出 {meta['out_ch']} 通道)")
    size = args.size or meta.get("size") or config.MAX_SIDE
    if args.size is None and meta.get("size"):
        print(f"输入长边 {size}（用模型训练时的设置）")
    elif meta.get("size") and args.size != meta.get("size"):
        print(f"[警告] 输入长边 {args.size} 与模型训练时（{meta['size']}）不一致："
              f"尺度不匹配会明显掉精度，建议按训练尺度跑")

    pairs = discover_pairs(args.data_dir)
    if not pairs:
        print(f"[错误] {args.data_dir} 下没有 图片+rsml 配对数据")
        sys.exit(1)

    mm = config.MM_PER_PX if args.mm_per_px is None else args.mm_per_px
    names = list(config.CLASS_NAMES)
    rows = []
    agg = {k: [] for k in ("gt_roots", "pred_roots", "gt_total", "pred_total")}
    per_ch_metrics = {n: {"iou": [], "dice": [], "accuracy": []} for n in names}
    t_start = time.time()
    for name, img_path, rsml_path in pairs:
        img = image_io.load_rgb(img_path)
        h0, w0 = img.shape[:2]
        roots = parse_rsml(rsml_path)
        gt_count, gt_lens, gt_total = root_stats(roots)
        w1, h1 = image_io.target_size(w0, h0, size, config.STRIDE)
        # GT 掩码与训练侧同一实现（三类通道），保证口径一致
        gt_masks, valid = build_target_masks(rsml_path, find_other(args.data_dir, name),
                                            (w0, h0), (w1, h1),
                                            config.MASK_LINE_WIDTH)
        gt = [np.ascontiguousarray(gt_masks[:, :, c]) for c in range(gt_masks.shape[2])]
        # 不变式：GT 根系也应限定在 GT 检查范围内（当前标注 100% 在框内，属校验性质）
        if valid[CH_CHECK] > 0:
            gt[CH_ROOT] = gt[CH_ROOT] & gt[CH_CHECK]

        res = predict.predict(model, img, max_side=size,
                              stride=config.STRIDE, device=device,
                              low_thresh=config.PRED_LOW_THRESHOLD)
        # 像素指标在**模型分辨率**上算，与训练时的验证 Dice 同一口径
        # （原图分辨率下 GT 是 5px 线、预测被上采样得较细，指标会被线宽差吃掉）
        preds = [res["probs"][c] > 0.5 for c in range(len(res["probs"]))]
        # 与部署同口径（inference.py 同样有 --no-anchor）：
        # 缺省把起点锚定到茎，补回被泡沫环挡住的那一段并计入根长
        st = analyze_mask_counted(
            res["mask_counted"], res["masks"][CH_STEM], anchor=args.anchor,
            spur=config.PRED_SPUR_LENGTH, min_len=config.MIN_ROOT_LENGTH,
            factor=config.STEM_ANCHOR_FACTOR, min_px=config.STEM_ANCHOR_MIN_PX,
            max_px=config.STEM_ANCHOR_MAX_PX)
        pred_cnt, pred_lens, pred_total = st["count"], st["lengths"], st["total"]

        ms = metrics.multi_channel_metrics(preds, gt, names=names, valid=valid)
        for m in ms:
            for k in ("iou", "dice", "accuracy"):
                per_ch_metrics[m["name"]][k].append(m[k])
        agg["gt_roots"].append(gt_count); agg["pred_roots"].append(pred_cnt)
        agg["gt_total"].append(gt_total); agg["pred_total"].append(pred_total)
        len_str = ";".join(f"{v:.1f}" for v in pred_lens[:30]) or "-"

        row = [name]
        for m in ms:
            row += [f"{m['iou']:.4f}" if not np.isnan(m["iou"]) else "-",
                    f"{m['dice']:.4f}" if not np.isnan(m["dice"]) else "-",
                    f"{m['accuracy']:.4f}" if not np.isnan(m["accuracy"]) else "-"]
        row += [gt_count, pred_cnt, st["anchored_count"],
                f"{gt_total:.1f}", f"{pred_total:.1f}", len_str]
        if mm and mm > 0:
            row += [f"{gt_total * mm:.1f}", f"{pred_total * mm:.1f}"]
        rows.append(row)

        desc = " | ".join(
            f"{m['name']} IoU={m['iou']:.4f} Dice={m['dice']:.4f}"
            if not np.isnan(m["iou"]) else f"{m['name']} 无真值" for m in ms)
        print(f"[{name}] {desc} | 根数 GT/预测 {gt_count}/{pred_cnt} "
              f"(起点锚定 {st['anchored_count']}) | "
              f"总长 GT/预测 {gt_total:.0f}/{pred_total:.0f}")

    el = time.time() - t_start

    def avg(k):
        return float(np.mean(agg[k]))

    def mae(a, b):
        return float(np.abs(np.asarray(agg[a]) - np.asarray(agg[b])).mean())

    header = ["图片名"]
    for n in names:
        header += [f"IoU({n})", f"Dice({n})", f"像素准确率({n})"]
    header += ["GT根数", "预测根数", "起点锚定(条)", "GT总长(px)", "预测总长(px)",
               "预测各根长(px,降序,至多30条)"]
    if mm and mm > 0:
        header += ["GT总长(mm)", "预测总长(mm)"]

    summary = [
        "",
        f"# ===== 汇总（{len(pairs)} 图平均） =====",
        f"# 根数: GT平均 {avg('gt_roots'):.1f} vs 预测平均 {avg('pred_roots'):.1f} "
        f"(平均绝对误差 {mae('gt_roots', 'pred_roots'):.2f} 根)",
        f"# 总长: GT平均 {avg('gt_total'):.0f} px vs 预测平均 {avg('pred_total'):.0f} px "
        f"(平均绝对误差 {mae('gt_total', 'pred_total'):.0f} px)",
    ]
    for n in names:
        dice = metrics.nanmean(per_ch_metrics[n]["dice"])
        iou = metrics.nanmean(per_ch_metrics[n]["iou"])
        acc = metrics.nanmean(per_ch_metrics[n]["accuracy"])
        summary.append(f"# {n}: Dice {dice:.4f} | IoU {iou:.4f} | 像素准确率 {acc:.4f}"
                       if not np.isnan(dice) else f"# {n}: 测试集无该通道真值")
    summary += [
        f"# 统计口径：输入长边 {size}；根系只在模型识别出的检查范围内统计，"
        + (f"且每条折线起点锚定到茎边界（补回被泡沫环挡住的那一段并计入根长）；"
           f"低阈值 {config.PRED_LOW_THRESHOLD} / 剪枝 {config.PRED_SPUR_LENGTH}px / "
           f"最短根 {config.MIN_ROOT_LENGTH}px / "
           f"锚定 {config.STEM_ANCHOR_FACTOR}×茎半径"
           f"({config.STEM_ANCHOR_MIN_PX:.0f}~{config.STEM_ANCHOR_MAX_PX:.0f}px)"
           if args.anchor else
           f"**起点不锚定**（--no-anchor，按可见根统计，适用于「只标看得见的根」的标注）；"
           f"低阈值 {config.PRED_LOW_THRESHOLD} / 剪枝 {config.PRED_SPUR_LENGTH}px / "
           f"最短根 {config.MIN_ROOT_LENGTH}px"),
        f"# 单位换算：1 px = {mm} mm" if mm else "# 未做 mm 换算",
        f"# 测试总耗时 {el:.1f}s | 单图平均 {el / max(len(pairs), 1):.2f}s",
    ]

    csv_path = naming.unique_path(pth.parent / f"model_test_{naming.timestamp()}.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        wr.writerows(rows)
        for line in summary:
            wr.writerow([line])

    print()
    for n in names:
        dice = metrics.nanmean(per_ch_metrics[n]["dice"])
        if not np.isnan(dice):
            print(f"{n}: Dice {dice:.4f} | IoU "
                  f"{metrics.nanmean(per_ch_metrics[n]['iou']):.4f} | 像素准确率 "
                  f"{metrics.nanmean(per_ch_metrics[n]['accuracy']):.4f}")
        else:
            print(f"{n}: 测试集无该通道真值")
    print(f"根数平均绝对误差 {mae('gt_roots', 'pred_roots'):.2f} 根 | "
          f"总长平均绝对误差 {mae('gt_total', 'pred_total'):.0f} px")
    print(f"测试总耗时 {el:.1f}s")
    print(f"结果已保存: {csv_path}")


if __name__ == "__main__":
    main()
