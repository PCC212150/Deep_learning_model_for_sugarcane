"""对任意图片文件夹做根系分割推理，统计 根数量 / 各根长度(px) / 总长度(px)，
并同时识别 茎横截面 与 检查范围(check_background)。

用法（readme 风格，两种写法均可）：
    python inference.py --model model_202609091135 --dir D:\\...\\某图片文件夹
    python inference.py --model_202609091135 "D:\\...\\某图片文件夹"   # 兼容
    python inference.py --dir D:\\...\\某图片文件夹                    # 模型省略=最新
    python inference.py --dir D:\\...\\某图片文件夹 --mm-per-px 0.1234  # CSV 追加 mm 列
    python inference.py --dir D:\\...\\某图片文件夹 --size 1536         # 覆盖输入长边

输入长边默认取**模型训练时的设置**（从权重里读），只有显式给 --size 才覆盖 ——
尺度必须与训练一致，否则精度会明显下降。

统计口径：
  - 根系只在**模型识别出的检查范围**内统计（范围外不计入），不扣茎；
  - 每条预测折线的**起点会锚定到茎边界**——茎外那圈黑色泡沫环不是根（模型判背景没错），
    但标注是从茎边开始画的，那一段被挡住、实际存在，所以补回来并计入根长。

结果：result/{目标文件夹名}/
    - {目标文件夹名}.csv    每行一张图（UTF-8 BOM，Excel 直接双击可开）：
                            图片名 根数量 起点锚定(条) 总根长(px) 平均根长(px) 最长根(px)
                            各根长度(px) 茎面积(px²) 检查区面积(px²) check_ok root_ok
                            「各根长度」用分号分隔；--mm-per-px>0 时行尾追加 mm 列；
                            文件末尾是若干以 # 开头的汇总行（Excel 可见，脚本可跳过）
    - {图片名}_overlay.png  原图 + 根系(红) + 茎(橙) + 检查范围(绿框)
    - {图片名}_mask.png     统计口径的根系掩码（已限定在检查范围内，黑底白根）
    - {图片名}_stem.png     识别出的茎横截面掩码
    - {图片名}_check.png    识别出的检查范围掩码
    - {图片名}.rsml         预测根系折线，每条折线一个 plant（不再分主根/侧根）
目录/文件重名时自动追加 -1、-2 …（项目规范）。
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from common import ckpt, image_io, naming, predict  # noqa: E402
from common.dataset import CH_CHECK, CH_ROOT, CH_STEM  # noqa: E402
from common.rsml_export import write_rsml  # noqa: E402
from common.skeleton_stats import analyze_mask_anchored  # noqa: E402


def parse_argv():
    """解析参数，兼容 readme 的 --model_xxx 与裸参数写法。"""
    model, folder, mm_per_px, size = None, None, None, None
    tokens = sys.argv[1:]
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "--model":
            model = tokens[i + 1] if i + 1 < len(tokens) else None
            i += 2
        elif t == "--mm-per-px":
            mm_per_px = float(tokens[i + 1]) if i + 1 < len(tokens) else None
            i += 2
        elif t == "--size":
            size = int(tokens[i + 1]) if i + 1 < len(tokens) else None
            i += 2
        elif t in ("--dir", "--image_dir", "--images", "--folder", "--path"):
            folder = tokens[i + 1] if i + 1 < len(tokens) else None
            i += 2
        elif t.startswith("--model_"):
            model = t[2:]
            i += 1
        elif t.startswith("--"):
            key = t[2:].lstrip("-")
            if "model" in key:
                model = tokens[i + 1] if i + 1 < len(tokens) else None
            else:
                folder = tokens[i + 1] if i + 1 < len(tokens) else None
            i += 2
        else:
            if model is None:
                model = t
            elif folder is None:
                folder = t
            else:
                print(f"[错误] 无法识别的参数: {t}")
                sys.exit(1)
            i += 1
    return model, folder, mm_per_px, size


def make_overlay(img: np.ndarray, masks, check_box, alpha: float = 0.45) -> np.ndarray:
    """把识别结果叠到原图上：根(红) + 茎(橙)，检查范围画绿框。

    单缓冲一次性合成（每张 5472x3648 图约 0.27s / 514MB 峰值；逐层转换会翻几倍）。
    """
    out = img.astype(np.float32)
    for ch, color in ((CH_ROOT, (255, 0, 0)), (CH_STEM, (255, 165, 0))):
        m = masks[ch]
        if m.any():
            out[m] = out[m] * (1.0 - alpha) + np.asarray(color, np.float32) * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)
    if check_box is not None:
        h, w = out.shape[:2]
        im = Image.fromarray(out)
        ImageDraw.Draw(im).rectangle(
            [check_box[0], check_box[1], check_box[2] - 1, check_box[3] - 1],
            outline=(0, 255, 0), width=max(2, int(min(w, h) * 0.004)))
        return np.asarray(im)
    return out


def main():
    model_arg, folder_arg, mm_arg, size_arg = parse_argv()
    if not folder_arg:
        print(__doc__)
        sys.exit(1)
    mm_per_px = config.MM_PER_PX if mm_arg is None else mm_arg

    img_dir = Path(folder_arg)
    if not img_dir.is_dir():
        print(f"[错误] 目标图片文件夹不存在: {img_dir}")
        sys.exit(1)

    pths, names = ckpt.resolve_pths(model_arg)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metas = ckpt.load_models(pths, device)     # 集成时 model 是模型列表
    meta = metas[0]
    tag = f"集成 {len(names)} 个" if len(names) > 1 else "模型"
    print(f"{tag}: {' + '.join(names)} | 设备: {device} | 输出 {meta['out_ch']} 通道"
          + (f" (epoch {meta['epoch']})" if meta.get("epoch") else ""))
    # 输入尺寸必须与训练时一致（实测：1024 训的模型用 2048 推理，总长误差从 4278px 涨到 9675px）
    size = size_arg or meta.get("size") or config.MAX_SIDE
    if size_arg is None and meta.get("size"):
        print(f"输入长边 {size}（用模型训练时的设置）")
    elif meta.get("size") and size_arg != meta.get("size"):
        print(f"[警告] 输入长边 {size_arg} 与模型训练时（{meta['size']}）不一致："
              f"尺度不匹配会明显掉精度，建议按训练尺度跑")

    imgs = sorted(p for p in img_dir.iterdir()
                  if p.suffix.lower() in config.IMAGE_EXTS)
    if not imgs:
        print(f"[错误] 目标文件夹里没有图片(支持 {sorted(config.IMAGE_EXTS)})")
        sys.exit(1)
    stems = [p.stem for p in imgs]
    dup = sorted({s for s in stems if stems.count(s) > 1})
    if dup:
        print(f"[错误] 文件夹里有同名不同扩展的图片 {dup}，输出文件会互相覆盖，"
              f"请先改名或分开处理")
        sys.exit(1)
    print(f"图片 {len(imgs)} 张: {img_dir}")

    # ---- 结果目录 result/{目标文件夹名}，重名追加 -1… ----
    out_dir = naming.unique_path(config.RESULT_DIR / img_dir.name)
    out_dir.mkdir(parents=True, exist_ok=False)
    csv_path = naming.unique_path(out_dir / f"{img_dir.name}.csv")
    mm = mm_per_px if mm_per_px and mm_per_px > 0 else 0.0

    header = ["图片名", "根数量", "起点锚定(条)", "总根长(px)", "平均根长(px)",
              "最长根(px)", "各根长度(px)", "茎面积(px²)", "检查区面积(px²)", "check_ok",
              "root_ok"]
    if mm:
        header += ["总根长(mm)", "平均根长(mm)", "最长根(mm)"]

    t_start = time.time()
    n_out = 5                 # 每张图输出：_mask/_stem/_check/_overlay.png + .rsml
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        for k, p in enumerate(imgs, 1):
            img = image_io.load_rgb(p)
            res = predict.predict(model, img, max_side=size,
                                  stride=config.STRIDE, device=device,
                                  low_thresh=config.PRED_LOW_THRESHOLD)
            masks = res["masks"]
            # 起点锚定到茎：补回被泡沫环挡住的那一段（计入根长，与标注同口径）
            st = analyze_mask_anchored(
                res["mask_counted"], masks[CH_STEM] if len(masks) > CH_STEM else None,
                spur=config.PRED_SPUR_LENGTH, min_len=config.MIN_ROOT_LENGTH,
                factor=config.STEM_ANCHOR_FACTOR, min_px=config.STEM_ANCHOR_MIN_PX,
                max_px=config.STEM_ANCHOR_MAX_PX)
            count, lens, total = st["count"], st["lengths"], st["total"]
            len_str = ";".join(f"{v:.1f}" for v in lens) if lens else "-"
            stem_area = int(masks[CH_STEM].sum()) if len(masks) > CH_STEM else 0
            check_area = (int((res["check_box"][2] - res["check_box"][0])
                              * (res["check_box"][3] - res["check_box"][1]))
                          if res["check_ok"] else img.shape[0] * img.shape[1])
            row = [p.name, count, st["anchored_count"], f"{total:.1f}",
                   f"{total / count:.1f}" if count else "0.0",
                   f"{max(lens):.1f}" if lens else "0.0", len_str,
                   stem_area, check_area, "是" if res["check_ok"] else "否",
                   "是" if res.get("root_ok", True) else "否"]
            if mm:
                row += [f"{total * mm:.1f}",
                        f"{total / count * mm:.1f}" if count else "0.0",
                        f"{max(lens) * mm:.1f}" if lens else "0.0"]
            wr.writerow(row)

            # ---- 保存识别结果图片 ----
            Image.fromarray((res["mask_counted"].astype(np.uint8) * 255)).save(
                out_dir / f"{p.stem}_mask.png")
            for ch, suffix in ((CH_STEM, "_stem"), (CH_CHECK, "_check")):
                m = masks[ch] if len(masks) > ch else np.zeros_like(res["mask_counted"])
                Image.fromarray((m.astype(np.uint8) * 255)).save(
                    out_dir / f"{p.stem}{suffix}.png")
            Image.fromarray(make_overlay(img, masks, res["check_box"])).save(
                out_dir / f"{p.stem}_overlay.png")

            # ---- 导出 RSML（每条折线一个 plant，不再分主根/侧根） ----
            rsml_path = write_rsml(out_dir / f"{p.stem}.rsml", file_key=p.stem,
                                   polylines=st["paths"])

            print(f"[{k}/{len(imgs)}] {p.name}: 根数 {count} | 总长 {total:.1f} px | "
                  f"各根长 {len_str[:60]}{'…' if len(len_str) > 60 else ''}")
            print(f"    检查范围 {'已识别' if res['check_ok'] else '未识别(全图统计)'} | "
                  f"茎面积 {stem_area} px² | 起点已锚定到茎 "
                  f"{st['anchored_count']}/{count} 条 | 已保存: {p.stem}_mask/_stem/"
                  f"_check/_overlay.png + .rsml")
        f.write(f"# 根系统计范围：模型识别出的检查范围（check_background），范围外不计入\n")
        f.write(f"# 起点锚定：每条预测折线的起点已补到茎边界，补回的那一段计入根长"
                f"（与标注口径一致）；「起点锚定(条)」是成功锚定的条数\n")
        f.write(f"# 单位：px（像素）；" + (f"mm 列按 1 px = {mm} mm 换算\n"
                                        if mm else "未做 mm 换算（--mm-per-px 关闭）\n"))
        f.write(f"# 参数：模型 {' + '.join(names)}，输入长边 {size}，"
                f"低阈值 {config.PRED_LOW_THRESHOLD}，剪枝 {config.PRED_SPUR_LENGTH}px，"
                f"最短根 {config.MIN_ROOT_LENGTH}px，"
                f"锚定阈值 {config.STEM_ANCHOR_FACTOR}×茎半径"
                f"（{config.STEM_ANCHOR_MIN_PX:.0f}~{config.STEM_ANCHOR_MAX_PX:.0f}px）\n")
        f.write(f"# 共 {len(imgs)} 张图，推理耗时 {time.time() - t_start:.1f}s\n")

    el = time.time() - t_start
    print(f"\n推理完成，总耗时 {el:.1f}s | 平均 {el / len(imgs):.2f}s/张")
    print(f"结果目录: {out_dir}")
    print(f"结果文件: {csv_path}")
    print(f"已保存文件: 每张图 {n_out} 个（图片名_mask/_stem/_check/_overlay.png + .rsml），"
          f"共 {len(imgs) * n_out} 个")


if __name__ == "__main__":
    main()
