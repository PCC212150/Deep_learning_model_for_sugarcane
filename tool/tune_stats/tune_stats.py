"""参数扫描：在带 RSML 真值的数据集上扫「掩码后处理」参数，按与真值的误差选最优。

统计口径（与 test.py 一致）：
    根总数、总长：来自 common.skeleton_stats（lengths 之和）；
    主根数/侧根数、主根总长/侧根总长：来自 common.root_hierarchy 的折线级父子推断。
真值侧同样按 主根/侧根 分开（label 与 ID 层级，见 root_hierarchy.gt_split）。

为什么快：掩码 -> 骨架 -> 邻接表这一步与阈值无关，同一张图只算一次（_skeleton_adj），
多组 spur/min_len 复用邻接表（_strands_from_adj 内部拷贝，不改入参）。

用法（项目根目录下运行，pcc 环境）：
    python tool\\tune_stats\\tune_stats.py --dir datasets\\test --dry-run   # 先看组合数与预计耗时
    python tool\\tune_stats\\tune_stats.py --dir datasets\\test --limit 2   # 快速试两张
    python tool\\tune_stats\\tune_stats.py --dir datasets\\test             # 全量
    python tool\\tune_stats\\tune_stats.py --dir datasets\\test --gt-mask   # 用真值掩码当输入（上限实验）
结果：result/tune_stats/tune_stats_{年月日时分}.txt（表格）与 .json（最优组合，可直接抄进 config.py）。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from common import image_io, naming, predict  # noqa: E402
from common.dataset import discover_pairs  # noqa: E402
from common.gt_mask import draw_mask_from_roots  # noqa: E402
from common.image_io import prob_to_orig_mask, prob_to_orig_mask_hysteresis  # noqa: E402
from common.root_hierarchy import (gt_summary, hierarchy_summary,  # noqa: E402
                                   infer_hierarchy)
from common.rsml_parse import parse_rsml  # noqa: E402
from common.skeleton_stats import _decimate, _skeleton_adj, _strands_from_adj  # noqa: E402
from common.unet import UNet  # noqa: E402

# 粗估耗时用的常数（本机 RTX 5060、5472x3648 图、长边 1024 实测；只为 --dry-run 给个量级）
_SEC_INFER = 0.4        # 每张图前向 1 次
_SEC_SKEL = 0.35        # 每个 (low, erode) 的 掩码->骨架->邻接表
_SEC_COMBO = 0.06       # 每个 (spur, min_len, normalize) 的 剪枝+分链+分层


def parse_args():
    p = argparse.ArgumentParser(description="扫描根系统计的后处理参数")
    p.add_argument("--dir", type=Path, required=True,
                   help="数据集目录（图片 + 同名 .rsml 真值），路径含空格要加引号")
    p.add_argument("--model", default=None, help="模型文件夹名（可省 model_ 前缀）；缺省用最新")
    p.add_argument("--low", default="0.05,0.10,0.20",
                   help="滞回低阈值候选，逗号分隔（高阈值固定 0.5）")
    p.add_argument("--spur", default="20,30,45,60",
                   help="骨架剪枝长度候选(px)，逗号分隔")
    p.add_argument("--min-len", default="20", help="最短根长候选(px)，逗号分隔")
    p.add_argument("--erode", default="1", help="腐蚀次数候选，逗号分隔")
    p.add_argument("--normalize-count", default="both",
                   help="计数归一：on/off/both（both = 两种口径都扫，默认）")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 张（按文件名排序），0=全部")
    p.add_argument("--gt-mask", action="store_true",
                   help="用真值折线画的掩码代替模型输出（上限实验，不需要模型）")
    p.add_argument("--size", type=int, default=config.MAX_SIDE, help="模型输入长边像素")
    p.add_argument("--spacing", type=float, default=50.0, help="折线抽稀间距(px)")
    p.add_argument("--sort", default="score",
                   choices=("score", "total", "primary", "secondary"),
                   help="排序依据：score=综合分，其余=单项误差")
    p.add_argument("--top", type=int, default=15, help="控制台只打印前 N 名（表格文件写全）")
    p.add_argument("--out", type=Path, default=config.RESULT_DIR / "tune_stats",
                   help="结果目录（重名自动加 -1）")
    p.add_argument("--dry-run", action="store_true", help="只打印组合数与预计耗时，不推理不写文件")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def _floats(s):
    return [float(v) for v in str(s).split(",") if v.strip()]


def _ints(s):
    return [int(v) for v in str(s).split(",") if v.strip()]


def resolve_model_dir(model_arg):
    """与 test.py 同规则：给了名字就找它，没给就取最新。"""
    root = config.MODEL_DIR
    if model_arg:
        cand = Path(model_arg) if Path(model_arg).is_absolute() else root / model_arg
        if not cand.exists():
            cand = root / f"model_{model_arg}"
        if not cand.exists():
            avail = sorted(p.name for p in root.glob("model_*") if p.is_dir())
            sys.exit(f"[错误] 找不到模型 {model_arg}。可用模型: {avail}")
        return cand
    dirs = [p for p in root.glob("model_*") if p.is_dir()]
    if not dirs:
        sys.exit(f"[错误] {root} 下没有模型，请先运行 train/train.py")
    # 按文件夹名取最新：model_YYYYMMDDHHMM 本身有序，而 mtime 会被写进目录的测试结果文件改掉
    return max(dirs, key=lambda p: p.name)


def load_model(model_arg, device):
    folder = resolve_model_dir(model_arg)
    pth = folder / f"{folder.name}.pth"
    if not pth.exists():
        pths = sorted(folder.glob("*.pth"))
        if not pths:
            sys.exit(f"[错误] {folder} 中没有 .pth 权重文件")
        pth = pths[-1]
    ckpt = torch.load(pth, map_location="cpu")
    model = UNet(in_ch=3, out_ch=1)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, folder


def score_of(e):
    """综合分：先对齐「根数」口径，长度做量级约束（长度单位是像素，除以 100 折算）。"""
    return (e["total"] + e["primary"] + e["secondary"]
            + (e["primary_len"] + e["secondary_len"]) / 100.0)


def main():
    args = parse_args()
    lows = _floats(args.low)
    spurs = _floats(args.spur)
    min_lens = _floats(args.min_len)
    erodes = _ints(args.erode)
    norms = [True, False] if args.normalize_count == "both" else \
        [args.normalize_count != "off"]
    combos = len(lows) * len(spurs) * len(min_lens) * len(erodes) * len(norms)

    pairs = discover_pairs(args.dir)
    if args.limit:
        pairs = pairs[:args.limit]
    if not pairs:
        sys.exit(f"[错误] {args.dir} 下没有「图片 + 同名 .rsml」配对数据")
    print(f"数据集: {args.dir} | 图片 {len(pairs)} 张" + (f"（limit={args.limit}）" if args.limit else ""))
    print(f"网格: low={lows} × spur={spurs} × min_len={min_lens} × erode={erodes} "
          f"× normalize={['on' if n else 'off' for n in norms]} = {combos} 组合")

    if args.dry_run:
        est = len(pairs) * (_SEC_COMBO * combos + _SEC_SKEL * len(lows) * len(erodes)
                            + (0 if args.gt_mask else _SEC_INFER))
        print(f"[dry-run] 预计耗时 ≈ {est / 60:.1f} 分钟"
              f"（模型前向 {len(pairs)} 次、骨架化 {len(pairs) * len(lows) * len(erodes)} 次）")
        print("[dry-run] 未推理、未写文件。去掉 --dry-run 即正式运行。")
        return

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = folder = None
    if not args.gt_mask:
        model, folder = load_model(args.model, device)
        print(f"模型: {folder.name} | 设备: {device}")
    else:
        print("上限实验：用真值折线画的掩码当输入（不加载模型）")

    keys = [("total", "根总数"), ("primary", "主根数"), ("secondary", "侧根数"),
            ("total_len", "总长"), ("primary_len", "主根长"), ("secondary_len", "侧根长")]
    errors = {k: {} for k, _ in keys}   # (low,spur,min_len,erode,norm) -> [逐图绝对误差]
    t0 = time.time()
    checked = False

    for si, (name, img_path, rsml_path) in enumerate(pairs, 1):
        img = image_io.load_rgb(img_path)
        h0, w0 = img.shape[:2]
        roots = parse_rsml(rsml_path)
        g = gt_summary(roots)
        gt_stats = {"total": len(roots), "primary": g["primary_count"],
                    "secondary": g["secondary_count"],
                    "total_len": g["total_length"],
                    "primary_len": g["primary_length"],
                    "secondary_len": g["secondary_length"]}
        if args.gt_mask:
            source_mask = draw_mask_from_roots(roots, (w0, h0), config.MASK_LINE_WIDTH)
            prob = None
        else:
            res = predict.predict(model, img, max_side=args.size, stride=config.STRIDE,
                                  device=device, low_thresh=0)
            # 概率图整张只前向一次，多组 low 阈值复用（predict 返回 numpy，阈值函数要 [1,1,h,w] 张量）
            prob = torch.from_numpy(res["prob_target"]).unsqueeze(0).unsqueeze(0)

        for low in lows:
            if args.gt_mask:
                masks = [source_mask]
            else:
                masks = [prob_to_orig_mask_hysteresis(prob, w0, h0, high=0.5, low=low)
                         if low > 0 else prob_to_orig_mask(prob, w0, h0, 0.5)]
            for erode in erodes:
                for m in masks:
                    adj = _skeleton_adj(m, erode)
                    for spur in spurs:
                        for min_len in min_lens:
                            for norm in norms:
                                key = (low, spur, min_len, erode, norm)
                                if adj is None:
                                    pred = {"total": 0, "primary": 0, "secondary": 0,
                                            "total_len": 0.0, "primary_len": 0.0,
                                            "secondary_len": 0.0}
                                else:
                                    entries = _strands_from_adj(adj, spur, min_len, norm)
                                    paths = [_decimate(e[1], args.spacing) for e in entries]
                                    s = hierarchy_summary(paths, infer_hierarchy(paths, spacing=args.spacing))
                                    pred = {"total": len(paths),
                                            "primary": s["primary_count"],
                                            "secondary": s["secondary_count"],
                                            "total_len": s["total_length"],
                                            "primary_len": s["primary_length"],
                                            "secondary_len": s["secondary_length"]}
                                for k, _ in keys:
                                    errors[k].setdefault(key, []).append(
                                        abs(pred[k] - gt_stats[k]))
                                # 首次组合做一次「自建路径 == analyze_mask_ex」的等价性自检
                                if not checked and adj is not None and not args.gt_mask:
                                    from common.skeleton_stats import analyze_mask_ex
                                    ref = analyze_mask_ex(m, spur=spur, min_len=min_len,
                                                          erode_iters=erode,
                                                          normalize_count=norm)
                                    if ref["count"] != len(paths) or \
                                            any(abs(a - b) > 1e-6 for a, b in
                                                zip(ref["lengths"], [e[0] for e in entries])):
                                        sys.exit("[错误] 骨架复用路径与 analyze_mask_ex 结果不一致，"
                                                 "请检查 common/skeleton_stats.py 是否被改动")
                                    checked = True
        print(f"  [{si}/{len(pairs)}] {name}: GT 根 {gt_stats['total']} "
              f"(主 {gt_stats['primary']} 侧 {gt_stats['secondary']}) "
              f"| 已用 {time.time() - t0:.0f}s")

    # ---- 汇总：每个组合的逐图平均绝对误差 ----
    rows = []
    for key in sorted(errors["total"], key=lambda k: (k[0], k[1], k[2], k[3], not k[4])):
        e = {k: float(np.mean(errors[k][key])) for k, _ in keys}
        e["score"] = score_of(e)
        rows.append((key, e))
    sort_key = {"score": "score", "total": "total",
                "primary": "primary", "secondary": "secondary"}[args.sort]
    rows.sort(key=lambda r: r[1][sort_key])

    out_dir = naming.unique_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=False)
    ts = naming.timestamp()
    txt = naming.unique_path(out_dir / f"tune_stats_{ts}.txt")
    best_key, best = rows[0]
    head = [f"# 数据集: {args.dir}   图片数 {len(pairs)}" + (f" (limit={args.limit})" if args.limit else ""),
            f"# 模型: {folder.name if folder else '（真值掩码实验）'} | 设备: {device} | 输入长边 {args.size}",
            f"# 排序: {args.sort} | 综合分 = 根总数MAE + 主根数MAE + 侧根数MAE + (主根长MAE+侧根长MAE)/100",
            f"# 网格: low={lows} spur={spurs} min_len={min_lens} erode={erodes} "
            f"normalize={['on' if n else 'off' for n in norms]} = {combos} 组合",
            "#",
            "# 排名  low  spur  min_len  erode  normalize | "
            + "  ".join(f"{cn}MAE" for _, cn in keys) + " | score"]
    lines = list(head)
    for i, (key, e) in enumerate(rows, 1):
        low, spur, min_len, erode, norm = key
        lines.append(f"{i:5d} {low:5.2f} {spur:6.0f} {min_len:8.0f} {erode:6d}  "
                     f"{'on' if norm else 'off':9s} | "
                     + "  ".join(f"{e[k]:8.2f}" for k, _ in keys) + f" | {e['score']:7.2f}")
    cur = {"low": config.PRED_LOW_THRESHOLD, "spur": config.PRED_SPUR_LENGTH,
           "min_len": config.MIN_ROOT_LENGTH, "normalize": True}
    cur_key = next((k for k in errors["total"]
                    if abs(k[0] - cur["low"]) < 1e-9 and abs(k[1] - cur["spur"]) < 1e-9
                    and abs(k[2] - cur["min_len"]) < 1e-9 and k[4] == cur["normalize"]), None)
    lines += ["", f"# 最优组合: --low {best_key[0]} --spur {best_key[1]:.0f} "
                  f"--min-len {best_key[2]:.0f} --erode {best_key[3]} "
                  f"--normalize-count {'on' if best_key[4] else 'off'}"]
    if cur_key is not None:
        ce = {k: float(np.mean(errors[k][cur_key])) for k, _ in keys}
        lines.append(f"# 与当前 config（low={cur['low']}, spur={cur['spur']:.0f}, "
                     f"min_len={cur['min_len']:.0f}, normalize=on）对比: "
                     f"根总数MAE {ce['total']:.2f} -> {best['total']:.2f}, "
                     f"侧根数MAE {ce['secondary']:.2f} -> {best['secondary']:.2f}")
    el = time.time() - t0
    lines.append(f"# 耗时: 总 {el:.1f}s | 单图 {el / len(pairs):.2f}s | 组合平均 {el / combos:.3f}s")
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    js = naming.unique_path(out_dir / f"tune_stats_{ts}.json")
    js.write_text(json.dumps({
        "low_thresh": best_key[0], "pred_spur_length": best_key[1],
        "min_root_length": best_key[2], "erode_iters": best_key[3],
        "normalize_count": best_key[4], "score": best["score"],
        "mae": {k: best[k] for k, _ in keys},
        "n_images": len(pairs), "data_dir": str(args.dir),
        "model": folder.name if folder else "gt-mask", "sort": args.sort,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n== 前 {min(args.top, len(rows))} 名（按 {args.sort}）==")
    print(f"{'排名':>4s} {'low':>5s} {'spur':>5s} {'min_len':>7s} {'erode':>5s} {'norm':>4s} | "
          + "  ".join(f"{cn+'MAE':>9s}" for _, cn in keys) + f" | {'score':>7s}")
    for i, (key, e) in enumerate(rows[:args.top], 1):
        print(f"{i:4d} {key[0]:5.2f} {key[1]:5.0f} {key[2]:7.0f} {key[3]:5d} "
              f"{'on' if key[4] else 'off':>4s} | "
              + "  ".join(f"{e[k]:9.2f}" for k, _ in keys) + f" | {e['score']:7.2f}")
    print(f"\n最优组合: low={best_key[0]} spur={best_key[1]:.0f} "
          f"min_len={best_key[2]:.0f} erode={best_key[3]} "
          f"normalize={'on' if best_key[4] else 'off'}")
    print(f"表格: {txt}")
    print(f"最优参数(json): {js}")


if __name__ == "__main__":
    main()
