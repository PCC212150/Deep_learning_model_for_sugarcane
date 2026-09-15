"""labelme 标注（labels/other/*.json）解析：茎横截面 + 检查范围。

json 结构（labelme 6.x）：
    {"version","flags","shapes":[{label,shape_type,points,...}],
     "imagePath","imageData","imageHeight","imageWidth"}

本项目只用两类 label：
    stem              甘蔗茎的横截面（polygon）
    check_background  框选检查的范围（rectangle，两点轴对齐）

解析结果统一为**原图坐标**下的矢量（点列 / 外接矩形），画掩码时按目标尺寸换算
（见 common/gt_mask.py），这样同一份标注可以按任意输入分辨率绘制。

注意：labelme 默认会把整张图 base64 塞进 imageData（本项目 29/31 个文件都有，单个最大 28MB），
解析后必须立刻丢弃，绝不能把整个 dict 缓存下来。
"""
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

STEM_LABEL = "stem"
CHECK_LABEL = "check_background"


@dataclass
class OtherLabels:
    """一张图的 labelme 标注（原图坐标）。"""

    path: Path = None
    stems: list = field(default_factory=list)       # [[(x, y), ...], ...] 茎多边形
    check_rect: tuple = None                        # (x0, y0, x1, y1) 检查范围外接矩形
    info: dict = field(default_factory=dict)        # 自检信息（shape 数 / 面积占比 / 告警）

    @property
    def ok(self) -> bool:
        """两类标注都解析到了才为 True（缺任一类的图，训练时会屏蔽对应通道的损失）。"""
        return bool(self.stems) and self.check_rect is not None


def _finite_points(points) -> list:
    """过滤非有限坐标（NaN/inf），返回 [(x, y), ...]。"""
    out = []
    for p in points or []:
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            out.append((x, y))
    return out


def _polygon_area(points) -> float:
    """鞋带公式算多边形面积（用于面积占比告警，不做精确统计）。"""
    if len(points) < 3:
        return 0.0
    s = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def parse_other(json_path, image_size=None, verbose: bool = True) -> OtherLabels:
    """解析 labelme json，返回 OtherLabels（原图坐标）。

    image_size: (w, h) 磁盘上原图的实际尺寸；给了就校验与 json 里记录的一致
                （标注画在别的尺寸上会导致掩码整体错位，必须报错而不是静默继续）。
    verbose: 打印异常/未知 label 的告警（每张图每类只打一次）。
    """
    json_path = Path(json_path)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    data.pop("imageData", None)          # 594MB 的坑：解析后立刻丢弃

    lab = OtherLabels(path=json_path)
    warns = []

    if image_size is not None:
        w, h = int(image_size[0]), int(image_size[1])
        jw, jh = data.get("imageWidth"), data.get("imageHeight")
        if jw is not None and jh is not None and (int(jw), int(jh)) != (w, h):
            raise ValueError(
                f"{json_path.name}: 标注尺寸 {int(jw)}x{int(jh)} 与图片实际尺寸 "
                f"{w}x{h} 不一致，掩码会整体错位。请重新导出/导出后再标注。")

    shapes = data.get("shapes") or []
    labels_seen = []
    for s in shapes:
        label = (s.get("label") or "").strip()
        labels_seen.append(label)
        pts = _finite_points(s.get("points"))
        if len(pts) != len(s.get("points") or []):
            warns.append(f"有 {len(s.get('points') or []) - len(pts)} 个非有限坐标点被丢弃")

        if label == STEM_LABEL:
            if len(pts) >= 3:
                lab.stems.append(pts)
            else:
                warns.append(f"stem 只有 {len(pts)} 个点，忽略")
        elif label == CHECK_LABEL:
            if not pts:
                warns.append("check_background 没有有效点，忽略")
                continue
            # 2 点/4 点/旋转矩形一律取外接矩形（当前标注全是 2 点轴对齐）
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            box = (min(xs), min(ys), max(xs), max(ys))
            lab.check_rect = box if lab.check_rect is None else _union(lab.check_rect, box)
        else:
            warns.append(f"未知 label {label!r}，已忽略")

    lab.info = {
        "n_shapes": len(shapes),
        "labels": labels_seen,
        "n_stem": len(lab.stems),
        "check_rect": lab.check_rect,
        "warns": warns,
    }
    if image_size is not None:
        w, h = int(image_size[0]), int(image_size[1])
        if lab.stems:
            area = sum(_polygon_area(p) for p in lab.stems)
            lab.info["stem_area_ratio"] = area / float(w * h)
        if lab.check_rect:
            x0, y0, x1, y1 = lab.check_rect
            lab.info["check_area_ratio"] = max(0.0, (x1 - x0)) * max(0.0, (y1 - y0)) / float(w * h)

    if verbose and warns:
        print(f"[标注告警] {json_path.name}: " + "；".join(sorted(set(warns))))
    return lab


def _union(a, b) -> tuple:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
