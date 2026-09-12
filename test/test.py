"""在测试集（datasets/test，每组含 png + rsml 真值）上评测已训练的模型。

用法（readme 两种写法均支持）：
    python test/test.py                                   # 自动使用最新模型
    python test/test.py --model model_202609091135
    python test/test.py --model_202609091135              # 兼容写法

输出：像素准确率 / IoU / Dice（掩码级），并附每张图"预测根数/总长 vs RSML 真值"汇总，
以及 主根/侧根 分开的条数与长度对比（侧根判据见 common/root_hierarchy.py）；
结果保存至模型文件夹内 model_test_{年月日时分}.txt（重名追加 -1）。
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from common import gt_mask, image_io, metrics, naming, predict  # noqa: E402
from common.root_hierarchy import (gt_summary, hierarchy_summary,  # noqa: E402
                                   infer_hierarchy)
from common.rsml_parse import parse_rsml  # noqa: E402
from common.skeleton_stats import analyze_mask_ex  # noqa: E402
from common.unet import UNet  # noqa: E402


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
    p = argparse.ArgumentParser(description="测试甘蔗根系 U-Net")
    p.add_argument("--model", default=None,
                   help="模型文件夹名(可省略 model_ 前缀)；缺省自动取最新")
    p.add_argument("--size", type=int, default=config.MAX_SIDE)
    p.add_argument("--data-dir", type=Path, default=config.TEST_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=config.MODEL_DIR)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args(preprocess_argv())


def resolve_model_dir(model_arg: str | None, root=None) -> Path:
    root = root or config.MODEL_DIR
    if model_arg:
        cand = Path(model_arg) if Path(model_arg).is_absolute() else root / model_arg
        if not cand.exists():
            cand = root / f"model_{model_arg}"  # 兼容省略前缀
        if not cand.exists():
            avail = sorted(p.name for p in root.glob("model_*") if p.is_dir())
            print(f"[错误] 找不到模型 {model_arg}。可用模型: {avail}")
            sys.exit(1)
        return cand
    dirs = [p for p in root.glob("model_*") if p.is_dir()]
    if not dirs:
        print(f"[错误] model 目录下没有模型，请先运行 train/train.py。{root}")
        sys.exit(1)
    # 按文件夹名取最新（model_YYYYMMDDHHMM 有序）；mtime 会被写进目录的测试结果文件改掉
    return max(dirs, key=lambda p: p.name)


def main():
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available()
                          else "cuda")
    folder = resolve_model_dir(args.model, args.out_dir)
    pth = folder / f"{folder.name}.pth"
    print(f"模型目录: {folder}")
    if not pth.exists():
        pths = sorted(folder.glob("*.pth"))
        if not pths:
            print(f"[错误] {folder} 中没有 .pth 权重文件")
            sys.exit(1)
        pth = pths[-1]

    ckpt = torch.load(pth, map_location="cpu")
    model = UNet(in_ch=3, out_ch=1)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    ckpt_epoch = ckpt.get("epoch", "?")
    ckpt_dice = ckpt.get("val_dice", None)
    print(f"权重: {pth.name} (保存于 epoch {ckpt_epoch}"
          + (f"，训练验证Dice {ckpt_dice:.4f}" if ckpt_dice else "") + ")")

    # ---- 逐图评测 ----
    from common.dataset import discover_pairs
    pairs = discover_pairs(args.data_dir)
    if not pairs:
        print(f"[错误] {args.data_dir} 下没有 png+rsml 配对数据")
        sys.exit(1)

    rows = []
    agg = {k: [] for k in ("iou", "dice", "acc", "gt_roots", "pred_roots",
                           "gt_total", "pred_total", "gt_prim", "pred_prim",
                           "gt_sec", "pred_sec", "gt_prim_len", "pred_prim_len",
                           "gt_sec_len", "pred_sec_len")}
    t_start = time.time()
    for name, img_path, rsml_path in pairs:
        img = image_io.load_rgb(img_path)
        h0, w0 = img.shape[:2]
        roots = parse_rsml(rsml_path)
        g = gt_summary(roots)          # 标注侧：主根/侧根 条数与长度
        w1, h1 = image_io.target_size(w0, h0, args.size, config.STRIDE)
        gt = image_io.resize_bool_mask(
            gt_mask.draw_mask_from_roots(roots, (w0, h0), config.MASK_LINE_WIDTH),
            w1, h1)

        res = predict.predict(model, img, max_side=args.size,
                              stride=config.STRIDE, device=device,
                              low_thresh=config.PRED_LOW_THRESHOLD)
        pb = res["prob_target"] > 0.5
        m = metrics.binary_metrics(pb, gt)
        st = analyze_mask_ex(res["mask_orig"], spur=config.PRED_SPUR_LENGTH,
                            min_len=config.MIN_ROOT_LENGTH, with_paths=True)
        # 预测侧：折线级几何判父子（端点长在别的折线上=侧根），口径见 common/root_hierarchy.py
        p = hierarchy_summary(st["paths"], infer_hierarchy(st["paths"]))
        pred_cnt, pred_lens, pred_total = st["count"], st["lengths"], st["total"]

        agg["iou"].append(m["iou"]); agg["dice"].append(m["dice"])
        agg["acc"].append(m["accuracy"])
        agg["gt_roots"].append(len(roots)); agg["pred_roots"].append(pred_cnt)
        agg["gt_total"].append(g["total_length"]); agg["pred_total"].append(pred_total)
        agg["gt_prim"].append(g["primary_count"]); agg["pred_prim"].append(p["primary_count"])
        agg["gt_sec"].append(g["secondary_count"]); agg["pred_sec"].append(p["secondary_count"])
        agg["gt_prim_len"].append(g["primary_length"]); agg["pred_prim_len"].append(p["primary_length"])
        agg["gt_sec_len"].append(g["secondary_length"]); agg["pred_sec_len"].append(p["secondary_length"])
        len_str = ",".join(f"{v:.1f}" for v in pred_lens[:30]) or "-"
        rows.append(f"{name}\t{m['iou']:.4f}\t{m['dice']:.4f}\t{m['accuracy']:.4f}"
                    f"\t{len(roots)}\t{pred_cnt}\t{g['total_length']:.1f}\t{pred_total:.1f}"
                    f"\t{g['primary_count']}\t{p['primary_count']}"
                    f"\t{g['secondary_count']}\t{p['secondary_count']}"
                    f"\t{g['primary_length']:.1f}\t{p['primary_length']:.1f}"
                    f"\t{g['secondary_length']:.1f}\t{p['secondary_length']:.1f}"
                    f"\t{len_str}")
        print(f"[{name}] IoU={m['iou']:.4f} Dice={m['dice']:.4f} "
              f"准确率={m['accuracy']:.4f} | 根数 GT/预测 {len(roots)}/{pred_cnt} "
              f"(主 {g['primary_count']}/{p['primary_count']}"
              f" 侧 {g['secondary_count']}/{p['secondary_count']}) | "
              f"总长 GT/预测 {g['total_length']:.0f}/{pred_total:.0f}")

    el = time.time() - t_start
    def avg(k):
        return float(np.mean(agg[k]))
    def mae(k):
        a, b = np.asarray(agg[f"gt_{k}"]), np.asarray(agg[f"pred_{k}"])
        return float(np.abs(a - b).mean())

    summary = [
        "",
        f"===== 汇总（{len(pairs)} 图平均） =====",
        f"像素准确率 {avg('acc'):.4f} | IoU {avg('iou'):.4f} | Dice {avg('dice'):.4f}",
        f"根数: GT平均 {avg('gt_roots'):.1f} vs 预测平均 {avg('pred_roots'):.1f} "
        f"(平均绝对误差 {mae('roots'):.2f} 根)",
        f"主根数: GT平均 {avg('gt_prim'):.1f} vs 预测平均 {avg('pred_prim'):.1f} "
        f"(平均绝对误差 {mae('prim'):.2f} 根)",
        f"侧根数: GT平均 {avg('gt_sec'):.1f} vs 预测平均 {avg('pred_sec'):.1f} "
        f"(平均绝对误差 {mae('sec'):.2f} 根)",
        f"总长: GT平均 {avg('gt_total'):.0f} px vs 预测平均 {avg('pred_total'):.0f} px "
        f"(平均绝对误差 {mae('total'):.0f} px)",
        f"主根总长平均绝对误差 {mae('prim_len'):.0f} px | "
        f"侧根总长平均绝对误差 {mae('sec_len'):.0f} px",
        f"测试总耗时 {el:.1f}s | 单图平均 {el / max(len(pairs), 1):.2f}s",
    ]
    header = ("# 图片名\tIoU\tDice\t像素准确率\tGT根数\t预测根数\tGT总长(px)"
              "\t预测总长(px)\tGT主根数\t预测主根数\tGT侧根数\t预测侧根数"
              "\tGT主根总长(px)\t预测主根总长(px)\tGT侧根总长(px)\t预测侧根总长(px)"
              "\t预测各根长(px,降序,至多30条)")
    txt = naming.unique_path(folder / f"model_test_{naming.timestamp()}.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for line in rows:
            f.write(line + "\n")
        for line in summary:
            f.write(line + "\n")
    print(f"\n平均: 像素准确率 {avg('acc'):.4f} | IoU {avg('iou'):.4f} | "
          f"Dice {avg('dice'):.4f}")
    print(f"根数平均绝对误差 {mae('roots'):.2f} 根 | "
          f"总长平均绝对误差 {mae('total'):.0f} px")
    print(f"主根数平均绝对误差 {mae('prim'):.2f} 根 | "
          f"侧根数平均绝对误差 {mae('sec'):.2f} 根")
    print(f"测试总耗时 {el:.1f}s")
    print(f"结果已保存: {txt}")


if __name__ == "__main__":
    main()
