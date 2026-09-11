"""对任意图片文件夹做根系分割推理，统计 根数量 / 各根长度(px) / 总长度(px)。

用法（readme 风格，两种写法均可）：
    python inference.py --model model_202609091135 --dir D:\\...\\某图片文件夹
    python inference.py --model_202609091135 "D:\\...\\某图片文件夹"   # 兼容
    python inference.py --dir D:\\...\\某图片文件夹                    # 模型省略=最新

结果：result/{目标文件夹名}/
    - {目标文件夹名}.txt    每行四列（空格分隔）：
                           图片名  根数量  各根系长度(逗号分隔,1位小数)  总根系长度(px)
    - {图片名}_mask.png     每张图的预测二值掩码（黑底白根）
    - {图片名}_overlay.png  每张图的原图 + 预测区域红色半透明叠加（便于目视检查）
    - {图片名}.rsml         每张图的预测根系折线，与标注同格式（可用 RootNav/
                           rsml-visualizer 打开；每条根 = 一个 plant 下的 primary 根）
目录/文件重名时自动追加 -1、-2 …（项目规范）。
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from common import image_io, naming, predict  # noqa: E402
from common.rsml_export import write_rsml  # noqa: E402
from common.skeleton_stats import analyze_mask_ex  # noqa: E402
from common.unet import UNet  # noqa: E402


def resolve_model_dir(model_arg) -> Path:
    root = config.MODEL_DIR
    if model_arg:
        cand = Path(model_arg) if Path(model_arg).is_absolute() else root / model_arg
        if not cand.exists():
            cand = root / f"model_{model_arg}"
        if not cand.exists():
            avail = sorted(p.name for p in root.glob("model_*") if p.is_dir())
            print(f"[错误] 找不到模型 {model_arg}。可用模型: {avail}")
            sys.exit(1)
        return cand
    dirs = [p for p in root.glob("model_*") if p.is_dir()]
    if not dirs:
        print(f"[错误] model 目录下没有模型，请先运行 train/train.py")
        sys.exit(1)
    return max(dirs, key=lambda p: p.stat().st_mtime)


def parse_argv():
    """解析参数，兼容 readme 的 --model_xxx 与裸参数写法。"""
    model, folder = None, None
    tokens = sys.argv[1:]
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "--model":
            model = tokens[i + 1] if i + 1 < len(tokens) else None
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
    return model, folder


def make_overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.45,
                 color=(255, 0, 0)) -> np.ndarray:
    """把预测掩码以半透明颜色叠加到原图上，返回 uint8 RGB 图（便于目视检查）。"""
    out = img.astype(np.float32)
    if mask is not None and mask.any():
        tint = np.asarray(color, dtype=np.float32)
        out[mask] = out[mask] * (1.0 - alpha) + tint * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    model_arg, folder_arg = parse_argv()
    if not folder_arg:
        print(__doc__)
        sys.exit(1)

    img_dir = Path(folder_arg)
    if not img_dir.is_dir():
        print(f"[错误] 目标图片文件夹不存在: {img_dir}")
        sys.exit(1)

    model_dir = resolve_model_dir(model_arg)
    pth = model_dir / f"{model_dir.name}.pth"
    if not pth.exists():
        pths = sorted(model_dir.glob("*.pth"))
        pth = pths[-1] if pths else None
    if pth is None or not pth.exists():
        print(f"[错误] 模型文件夹中没有 .pth 权重: {model_dir}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_ch=3, out_ch=1)
    ckpt = torch.load(pth, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    print(f"模型: {model_dir.name} | 权重: {pth.name} | 设备: {device}")

    imgs = sorted(p for p in img_dir.iterdir()
                  if p.suffix.lower() in config.IMAGE_EXTS)
    if not imgs:
        print(f"[错误] 目标文件夹里没有图片(支持 {sorted(config.IMAGE_EXTS)})")
        sys.exit(1)
    print(f"图片 {len(imgs)} 张: {img_dir}")

    # ---- 结果目录 result/{目标文件夹名}，重名追加 -1… ----
    out_dir = naming.unique_path(config.RESULT_DIR / img_dir.name)
    out_dir.mkdir(parents=True, exist_ok=False)
    out_txt = naming.unique_path(out_dir / f"{img_dir.name}.txt")

    t_start = time.time()
    with open(out_txt, "w", encoding="utf-8") as f:
        for k, p in enumerate(imgs, 1):
            img = image_io.load_rgb(p)
            res = predict.predict(model, img, max_side=config.MAX_SIDE,
                                  stride=config.STRIDE, device=device,
                                  low_thresh=config.PRED_LOW_THRESHOLD)
            st = analyze_mask_ex(res["mask_orig"], spur=config.PRED_SPUR_LENGTH,
                                 min_len=config.MIN_ROOT_LENGTH, with_paths=True)
            count, lens, total = st["count"], st["lengths"], st["total"]
            len_str = ",".join(f"{v:.1f}" for v in lens) if lens else "-"
            row = f"{p.name} {count} {len_str} {total:.1f}"
            f.write(row + "\n")

            # ---- 保存处理过程中的图片：预测掩码 + 原图叠加可视化 ----
            mask_img = Image.fromarray(
                (res["mask_orig"].astype(np.uint8) * 255))
            mask_path = out_dir / f"{p.stem}_mask.png"
            mask_img.save(mask_path)
            overlay_path = out_dir / f"{p.stem}_overlay.png"
            Image.fromarray(make_overlay(img, res["mask_orig"])).save(overlay_path)

            # ---- 导出 RSML（与标注同格式的预测根系折线） ----
            rsml_path = write_rsml(out_dir / f"{p.stem}.rsml", file_key=p.stem,
                                   polylines=st["paths"])

            print(f"[{k}/{len(imgs)}] {p.name}: 根数 {count} | "
                  f"总长 {total:.1f} px | 各根长 {len_str}")
            print(f"    已保存: {mask_path.name} | {overlay_path.name} | "
                  f"{rsml_path.name}")
    el = time.time() - t_start
    print(f"\n推理完成，总耗时 {el:.1f}s | 平均 {el / len(imgs):.2f}s/张")
    print(f"结果目录: {out_dir}")
    print(f"结果文件: {out_txt}")
    print(f"已保存文件: 每张图 3 个（图片名_mask.png 预测掩码 / "
          f"图片名_overlay.png 原图叠加 / 图片名.rsml 预测折线），共 {len(imgs) * 3} 个")


if __name__ == "__main__":
    main()
