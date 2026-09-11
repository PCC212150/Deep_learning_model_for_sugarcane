"""把 RSML 解析出的根折线画成二值分割掩码（真值 GT）。

规范：折线在**原图分辨率**上以 5px 线宽绘制，之后与图像一起缩放到模型输入尺寸。
"""
import numpy as np
from PIL import Image, ImageDraw

from common.rsml_parse import Root


def draw_mask_from_roots(roots, image_size, width: int = 5) -> np.ndarray:
    """在原图尺寸 (w, h) 画二值掩码，返回 bool (h, w)。

    多根重叠处取并集；忽略点数 < 2 的根；越界线段由 PIL 自动裁剪。
    """
    w, h = image_size
    canvas = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(canvas)
    for r in roots:
        if len(r.points) < 2:
            continue
        pts = [(int(round(x)), int(round(y))) for x, y in r.points]
        draw.line(pts, fill=255, width=int(width), joint="curve")
    return np.asarray(canvas, dtype=np.uint8) > 0


def draw_mask_from_polylines(polyline_groups, image_size, width: int = 5) -> np.ndarray:
    """draw_mask_from_roots 的通用版本：直接给若干折线（每根一段列表）。"""
    roots = [Root(points=pts) for pts in polyline_groups if len(pts) >= 2]
    return draw_mask_from_roots(roots, image_size, width)
