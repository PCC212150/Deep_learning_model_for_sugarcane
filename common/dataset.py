"""PyTorch Dataset：png 与 rsml 同名配对，构造训练/验证数据。

- 每张图：原图画 5px 掩码 -> 与图像同参数缩放到 (max_side, 16 倍数)；
- 数据增强（仅训练集）：随机水平翻转、±10° 旋转、亮度/对比度抖动；
- 图像与掩码使用同一随机变换，保证对齐。
"""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from common import gt_mask, image_io
from common.rsml_parse import parse_rsml


def discover_pairs(data_dir, image_exts=None):
    """返回 [(stem, 图片路径, rsml路径), ...]：图片必须有同名 .rsml 配对。"""
    if image_exts is None:
        from config import IMAGE_EXTS
        image_exts = IMAGE_EXTS
    data_dir = Path(data_dir)
    pairs = []
    for img_path in sorted(p for p in data_dir.iterdir() if p.suffix.lower() in image_exts):
        rsml_path = img_path.with_suffix(".rsml")
        if rsml_path.exists():
            pairs.append((img_path.stem, img_path, rsml_path))
    return pairs


def _corner_fill(img: np.ndarray, patch: int = 32) -> tuple:
    """取图像四角小块的均值颜色，作为旋转增强的填充色（避免黑边假象）。"""
    h, w = img.shape[:2]
    corners = [img[0:patch, 0:patch], img[0:patch, w - patch:w],
               img[h - patch:h, 0:patch], img[h - patch:h, w - patch:w]]
    mean = np.mean(np.concatenate(corners).reshape(-1, 3), axis=0)
    return (int(mean[0]), int(mean[1]), int(mean[2]))


def _rotate_rgb(arr: np.ndarray, angle: float, fill: tuple) -> np.ndarray:
    im = Image.fromarray(arr).rotate(angle, resample=Image.Resampling.BILINEAR,
                                     fillcolor=fill)
    return np.asarray(im, dtype=np.uint8)


def _rotate_mask(mask: np.ndarray, angle: float) -> np.ndarray:
    im = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).rotate(
        angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
    return np.asarray(im, dtype=np.uint8) > 100


def _jitter(img: np.ndarray) -> np.ndarray:
    f = np.float32
    out = img.astype(np.float32)
    b = random.uniform(0.85, 1.15)
    c = random.uniform(0.85, 1.15)
    out = (out * b - 127.5) * c + 127.5
    return np.clip(out, 0, 255)


class RootDataset(Dataset):
    """逐项返回 (img[1,3,H,W] float32 0~1, gt[1,H,W] float32 0/1, name)。

    构造时完成解码/画掩码/缩放（较慢），训练时仅做轻量增强。
    """

    def __init__(self, data_dir, names=None, max_side=1024, stride=16,
                 mask_width=5, augment=False, seed=0):
        self.augment = augment
        pairs = discover_pairs(data_dir)
        if names is not None:
            wanted = set(names)
            pairs = [p for p in pairs if p[0] in wanted]
        self.names = [p[0] for p in pairs]

        self.items = []
        for name, img_path, rsml_path in pairs:
            img = image_io.load_rgb(img_path)
            h0, w0 = img.shape[:2]
            w1, h1 = image_io.target_size(w0, h0, max_side, stride)
            roots = parse_rsml(rsml_path)
            mask = gt_mask.draw_mask_from_roots(roots, (w0, h0), width=mask_width)
            self.items.append({
                "name": name,
                "img": image_io.resize_rgb(img, w1, h1),       # (h1,w1,3) uint8
                "mask": image_io.resize_bool_mask(mask, w1, h1),  # (h1,w1) bool
                "fill": _corner_fill(img),
                "n_roots": len(roots),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = it["img"]
        m = it["mask"]
        if self.augment:
            if random.random() < 0.5:
                img = np.flip(img, axis=1)
                m = np.flip(m, axis=1)
            angle = random.uniform(-10.0, 10.0)
            if abs(angle) > 0.3:
                img = _rotate_rgb(img, angle, it["fill"])
                m = _rotate_mask(m, angle)
            img = _jitter(img)
        img = np.ascontiguousarray(img)
        x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        y = torch.from_numpy(np.ascontiguousarray(m, dtype=np.float32)).unsqueeze(0)
        return x, y, it["name"]
