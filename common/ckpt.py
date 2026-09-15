"""模型权重的统一加载：从 state_dict 识别输出通道数，构造对应的 U-Net。

原来 inference.py / test.py / tool/tune_stats / experimental_report 各自写了一遍
`UNet(in_ch=3, out_ch=1)` + `load_state_dict`，改通道数时四处都要动，容易漏。这里统一。

通道数从 `state_dict["out.weight"]` 的形状读取（不依赖 hparams，旧权重也认得）。
"""
from pathlib import Path

import torch

from common.unet import UNet

# 当前流程要求的输出通道数（见 config.CLASS_NAMES：root / stem / check）
REQUIRED_CHANNELS = 3


def load_unet(pth, device="cpu", require: int = REQUIRED_CHANNELS):
    """加载权重并构造 U-Net，返回 (model, meta)。

    require 不为 None 时校验通道数：旧版的单通道（只有根系）权重会明确报错退出，
    而不是「静默降级成只输出根系」——后者会让茎/检查范围的统计悄悄失效。
    需要跑旧权重时传 require=None，并自行忽略多出来的通道。
    """
    pth = Path(pth)
    # weights_only=False：ckpt 里除权重外还带 hparams（可能含 numpy 标量等非张量对象），
    # torch>=2.6 默认的 weights_only=True 会拒绝加载。这里的文件都是本项目自己训练产出的。
    ckpt = torch.load(pth, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise SystemExit(f"[错误] {pth.name} 不是本项目 train.py 保存的权重"
                         f"（缺少 state_dict 字段）")
    sd = ckpt["state_dict"]
    if "out.weight" not in sd:
        raise SystemExit(f"[错误] {pth.name} 的 state_dict 里没有 out.weight，"
                         f"无法判断输出通道数")
    out_ch = int(sd["out.weight"].shape[0])
    if require is not None and out_ch != require:
        raise SystemExit(
            f"[错误] 权重 {pth.name} 是 {out_ch} 通道的旧模型，当前流程需要 "
            f"{require} 通道（根系 / 茎横截面 / 检查范围）。请用改造后重新训练得到的权重。")

    # 归一化类型必须与训练时一致（BatchNorm 与 GroupNorm 的参数形状不同，装错会加载失败）
    norm = ckpt.get("norm") or (ckpt.get("hparams") or {}).get("norm") or "batch"
    model = UNet(in_ch=3, out_ch=out_ch, norm=norm)
    model.load_state_dict(sd)
    model.to(device)
    hp = ckpt.get("hparams") or {}
    meta = {
        "out_ch": out_ch,
        "norm": norm,
        "epoch": ckpt.get("epoch"),
        "val_dice": ckpt.get("val_dice"),
        "class_names": ckpt.get("class_names"),
        "hparams": hp,
        # 训练时的输入长边：推理/测试要按它来，尺度不一致会明显掉精度（实测 1024 训的模型
        # 用 2048 推理，总长误差从 4278px 涨到 9675px）
        "size": hp.get("size"),
    }
    return model, meta
