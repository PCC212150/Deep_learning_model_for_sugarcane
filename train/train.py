"""训练 U-Net 分割甘蔗根系（多标签三通道：根系 / 茎横截面 / 检查范围）。

运行（pcc 环境，建议在项目根目录）：
    conda activate pcc
    python train/train.py                        # 默认 200 轮 / 早停 60
    python train/train.py --size 1536 --batch 8  # 部署到 4xRTX3090 服务器时

每轮输出：轮次 / 损失 / 验证Dice（根系通道） / 本轮耗时(s)（控制台 + 模型文件夹内日志），
行尾另附茎与检查范围通道的 Dice。模型保存：model/model_YYYYMMDDHHMM/
（验证最优轮权重 .pth + 参数日志 .txt + hparams.json）

数据划分**按植株整组进出**（同一植株的不同时点不会分处训练/验证两侧），
`--val-size` 是验证集**植株数**。
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from common import naming  # noqa: E402
from common.dataset import RootDataset, plant_key  # noqa: E402
from common.unet import UNet  # noqa: E402

N_CH = len(config.CLASS_NAMES)


def dice_loss(prob, target):
    """prob: sigmoid 后概率 (B,C,H,W), target: 0/1。返回逐样本逐通道 Dice 损失 (B,C)。"""
    eps = 1.0
    inter = (prob * target).sum(dim=(2, 3))
    den = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1.0 - (2.0 * inter + eps) / (den + eps)


def channel_weights(base, valid):
    """(C,) 基准权重 × (B,C) 通道有效性 -> (B,C)；缺标注的通道权重置 0（不参与损失）。"""
    w = torch.as_tensor(base, dtype=torch.float32,
                        device=valid.device).view(1, -1)
    return w * valid


def parse_args():
    p = argparse.ArgumentParser(description="训练甘蔗根系 U-Net（三通道多标签）")
    p.add_argument("--size", type=int, default=config.MAX_SIDE, help="输入长边像素")
    p.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    p.add_argument("--accum", type=int, default=config.ACCUM,
                   help="梯度累积步数：等效 batch = --batch × --accum。显存不够时用它换等效 batch"
                        "（默认见 config.ACCUM）")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--patience", type=int, default=config.PATIENCE,
                   help="验证 Dice 连续 N 轮无提升则早停（应 > --lr-patience）")
    p.add_argument("--lr-patience", type=int, default=config.LR_PATIENCE,
                   help="验证 Dice 连续 N 轮无提升则 LR 减半（默认见 config.LR_PATIENCE）")
    p.add_argument("--val-size", type=int, default=config.VAL_SIZE,
                   help="验证集植株数（同植株的全部时点整组进同一侧）")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--data-dir", type=Path, default=config.TRAIN_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=config.MODEL_DIR)
    p.add_argument("--norm", default=config.NORM, choices=("group", "batch"),
                   help="归一化层：group=GroupNorm（小 batch 稳定）、batch=BatchNorm"
                        "（需 batch≥2，默认见 config.NORM）")
    p.add_argument("--workers", type=int, default=config.NUM_WORKERS,
                   help="DataLoader 子进程数（0=主进程里同步加载）。服务器上设 4~8 可让"
                        "读图/增强与训练并行，避免 GPU 空等（默认见 config.NUM_WORKERS）")
    p.add_argument("--no-amp", action="store_true", help="关闭混合精度")
    p.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    return p.parse_args()


def split_by_plant(names, val_size, seed):
    """按植株分组划分：返回 (训练名列表, 验证名列表, 验证植株列表)。

    同一植株的所有时点整组进同一侧，避免「同株不同时点」跨训练/验证造成泄漏。
    val_size = 验证集植株数；config.VAL_PLANTS 非空时优先按它钉死。
    """
    groups = {}
    for n in names:
        groups.setdefault(plant_key(n), []).append(n)
    keys = sorted(groups)
    if config.VAL_PLANTS:
        pinned = [k for k in keys if k in set(config.VAL_PLANTS)]
        missing = sorted(set(config.VAL_PLANTS) - set(pinned))
        if missing:
            print(f"[警告] config.VAL_PLANTS 里的植株不在数据集中: {missing}")
        val_plants = pinned
    else:
        rng = np.random.RandomState(seed)
        rng.shuffle(keys)
        val_plants = sorted(keys[:max(val_size, 0)])
    val_set = set(val_plants)
    val_names = sorted(n for k in val_plants for n in groups[k])
    train_names = sorted(n for k, v in groups.items() if k not in val_set for n in v)
    return train_names, val_names, val_plants


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available()
                          else "cuda")
    if device.type == "cpu":
        print("[警告] 使用 CPU 训练，速度很慢。pcc 环境支持 CUDA（RTX 5060）。")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}  显存: "
              f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")

    # ---- 数据划分：按植株整组进出（保证可复现） ----
    names = [n for n, _, _ in _pairs(args.data_dir)]
    train_names, val_names, val_plants = split_by_plant(names, args.val_size,
                                                        args.seed)
    train_plants = sorted({plant_key(n) for n in train_names})
    print(f"数据: 共 {len(names)} 组 / {len(set(map(plant_key, names)))} 植株 | "
          f"训练 {len(train_names)} 组({len(train_plants)} 植株) | "
          f"验证 {len(val_names)} 组({len(val_plants)} 植株)")
    print(f"验证植株: {', '.join(val_plants) if val_plants else '无(不早停,保存最后轮)'}")
    t0 = time.time()
    train_ds = RootDataset(args.data_dir, names=train_names,
                           max_side=args.size, augment=True, seed=args.seed)
    val_ds = RootDataset(args.data_dir, names=val_names,
                         max_side=args.size, augment=False, seed=args.seed)
    assert len(train_ds) == len(train_names), "训练集样本数不符（名字对不上？）"
    assert len(val_ds) == len(val_names), "验证集样本数不符（名字对不上？）"
    print(f"数据加载完成，用时 {time.time() - t0:.1f}s")

    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
        persistent_workers=args.workers > 0)
    if args.workers == 0 and device.type == "cuda":
        print("[提示] num_workers=0：读图与数据增强在主进程里同步做，GPU 会空等。"
              "服务器上可加 --workers 8（本机 Windows 保持 0 即可）。")

    # ---- 模型 ----
    if args.norm == "batch" and args.batch < 2:
        print(f"[警告] 归一化用 BatchNorm 但 batch={args.batch}：batch=1 时读到的统计量"
              f"与推理用的滑动平均对不上，会出现严重欠分割。请用 batch≥2 或把 "
              f"config.NORM 改成 'group'。")
    model = UNet(in_ch=3, out_ch=N_CH, norm=args.norm).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    eff_batch = args.batch * args.accum
    print(f"U-Net 参数量: {n_params / 1e6:.2f}M | 输入长边 {args.size} | "
          f"batch {args.batch}" + (f"×累积{args.accum}={eff_batch}" if args.accum > 1 else "")
          + f" | 输出 {N_CH} 通道 {config.CLASS_NAMES}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=config.WEIGHT_DECAY)
    # 按验证指标自动降 LR。不用 CosineAnnealingLR(T_max=args.epochs)：T_max 是「上限轮数」
    # 而不是「实际会跑多少轮」，而早停总在上限之前触发，余弦因此永远走不到 —— 见 config.LR_PATIENCE
    # 的注释（实测全程恒定 1e-3，val 卡住不动）。
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config.LR_FACTOR,
        patience=args.lr_patience, min_lr=config.MIN_LR)
    cur_lr = args.lr
    amp = (device.type == "cuda") and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    # 逐通道正样本加权，(1,C,1,1) 对应 (B,C,H,W) 的通道维（见 config.LOSS_POS_WEIGHT）
    pos_w = torch.as_tensor(config.LOSS_POS_WEIGHT, dtype=torch.float32,
                            device=device).view(1, N_CH, 1, 1)

    # ---- 输出目录：model/model_年月日时分，重名追加 -1/-2… ----
    ts = naming.timestamp()
    # create_unique_dir 而非 unique_path：同一分钟内启动两个训练（一张卡一个）时，
    # 「先查后建」的写法会让两个进程拿到同一个名字、其中一个直接崩；这里是原子创建。
    folder = naming.create_unique_dir(args.out_dir, naming.model_folder_name(ts))
    ckpt_path = folder / f"{folder.name}.pth"
    log_path = folder / f"{folder.name}_log.txt"
    hparams = {k: (str(v) if isinstance(v, Path) else v)
               for k, v in vars(args).items()}
    hparams.update({"device": str(device), "gpu": torch.cuda.get_device_name(0)
                    if device.type == "cuda" else "cpu",
                    "val_names": val_names, "train_names": train_names,
                    "val_plants": val_plants, "train_plants": train_plants,
                    "class_names": list(config.CLASS_NAMES), "out_ch": N_CH,
                    "norm": args.norm,
                    "loss_bce_w": list(config.LOSS_BCE_W),
                    "loss_dice_w": list(config.LOSS_DICE_W),
                    "loss_pos_weight": list(config.LOSS_POS_WEIGHT),
                    "params_M": round(n_params / 1e6, 2),
                    "start": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model_name": folder.name})
    with open(folder / "hparams.json", "w", encoding="utf-8") as f:
        json.dump(hparams, f, ensure_ascii=False, indent=2)

    def log(msg, console=True):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        if console:
            print(msg)

    log(f"===== 训练开始 {hparams['start']} =====\n"
        f"hparams: {json.dumps(hparams, ensure_ascii=False)}", console=False)
    log(f"[信息] {hparams['gpu']} | 参数量 {n_params/1e6:.2f}M | "
        f"输入 {args.size} | batch {args.batch}"
        + (f"×累积{args.accum}={eff_batch}" if args.accum > 1 else "")
        + f" | lr {args.lr} | "
        f"epochs {args.epochs} | 训练 {len(train_names)} 组 | "
        f"验证 {len(val_names)} 组({len(val_plants)} 植株)")
    log(f"[信息] 通道 {config.CLASS_NAMES} | BCE权重 {config.LOSS_BCE_W} | "
        f"Dice权重 {config.LOSS_DICE_W} | 正样本权重 {config.LOSS_POS_WEIGHT}")
    log(f"[信息] 学习率 {args.lr} | 平台期 {args.lr_patience} 轮不减半就 ×{config.LR_FACTOR}"
        f"（下限 {config.MIN_LR}）| 早停 {args.patience} 轮")
    log(f"[信息] 模型目录: {folder}")

    # ---- 训练循环 ----
    best_val_dice, best_epoch = -1.0, -1
    bad_epochs = 0
    epoch, val_dice = 0, -1.0
    t_start = time.time()
    try:
        for epoch in range(1, args.epochs + 1):
            t_ep = time.time()
            model.train()
            loss_sum, n_batch = 0.0, 0
            optimizer.zero_grad(set_to_none=True)
            n_micro = 0
            for i_batch, (x, y, _, valid) in enumerate(loader):
                x, y, valid = x.to(device), y.to(device), valid.to(device)
                with torch.autocast(device_type="cuda", enabled=amp):
                    out = model(x)
                    prob = torch.sigmoid(out)
                    # 逐通道 BCE（缺标注的通道权重为 0）：用 reduction="none" 再按通道
                    # 加权，避免直接对 (B,C,H,W) 求 mean 时被大面积的检查范围通道带偏。
                    # pos_weight 必须是 (1,C,1,1) 才能对 (B,C,H,W) 正确广播到逐通道
                    # （写成 (C,) 或 (C,1,1) 会广播到 batch 维，静默算错）。
                    bce = F.binary_cross_entropy_with_logits(
                        out.float(), y, pos_weight=pos_w,
                        reduction="none").mean(dim=(2, 3))
                    wb = channel_weights(config.LOSS_BCE_W, valid)
                    loss_bce = (bce * wb).sum() / wb.sum().clamp(min=1e-6)
                    wd = channel_weights(config.LOSS_DICE_W, valid)
                    loss_dice = ((dice_loss(prob.float(), y) * wd).sum()
                                 / wd.sum().clamp(min=1e-6))
                    loss = loss_bce + loss_dice
                # 梯度累积：把 loss 按累积步数缩放，攒够 accum 个 micro-batch 再更新一次
                scaler.scale(loss / args.accum).backward()
                n_micro += 1
                if n_micro >= args.accum or (i_batch + 1) == len(loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    n_micro = 0
                loss_sum += float(loss.item())      # 记录未缩放的损失，便于跨配置比较
                n_batch += 1
            train_loss = loss_sum / max(n_batch, 1)

            # ---- 验证（轮内）：逐通道 Dice/IoU，早停只看根系通道 ----
            val_dice = val_iou = -1.0
            per_ch_dice = [-1.0] * N_CH
            if val_ds:
                model.eval()
                d_all, i_all = [], []
                with torch.no_grad():
                    for x, y, _, valid in val_loader:
                        x = x.to(device)
                        with torch.autocast(device_type="cuda", enabled=amp):
                            prob = torch.sigmoid(model(x)).float()
                        pb = prob.cpu().numpy() > 0.5
                        gt = y.numpy() > 0.5
                        vd = valid.numpy()
                        for i in range(len(gt)):
                            ds, is_ = [], []
                            for c in range(N_CH):
                                if vd[i, c] < 0.5:      # 该图缺这个通道的标注
                                    ds.append(np.nan)
                                    is_.append(np.nan)
                                    continue
                                tp = (pb[i, c] & gt[i, c]).sum()
                                d = 2.0 * tp / (pb[i, c].sum() + gt[i, c].sum() + 1e-8)
                                iou = tp / (pb[i, c].sum() + gt[i, c].sum() - tp + 1e-8)
                                ds.append(float(d))
                                is_.append(float(iou))
                            d_all.append(ds)
                            i_all.append(is_)
                if d_all:
                    with np.errstate(invalid="ignore"):
                        per_ch_dice = list(np.nanmean(np.asarray(d_all), axis=0))
                        per_ch_iou = list(np.nanmean(np.asarray(i_all), axis=0))
                    # 整个验证集都没有该通道标注时 nanmean 会返回 nan，统一记成 -1；
                    # 一律转成 Python float：numpy 标量写进 ckpt 会让 torch.load 的
                    # weights_only 模式（torch>=2.6 默认）拒绝加载
                    per_ch_dice = [float(-1.0 if np.isnan(v) else v) for v in per_ch_dice]
                    per_ch_iou = [float(-1.0 if np.isnan(v) else v) for v in per_ch_iou]
                    val_dice, val_iou = per_ch_dice[0], per_ch_iou[0]

            improved = val_dice - best_val_dice > 1e-4
            if improved:
                best_val_dice, best_epoch = val_dice, epoch
                torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                            "val_dice": val_dice, "hparams": hparams,
                            "out_ch": N_CH, "norm": args.norm,
                            "class_names": list(config.CLASS_NAMES)},
                           ckpt_path)
                bad_epochs = 0
            else:
                bad_epochs += 1

            dt = time.time() - t_ep
            note = ""
            if val_ds:
                # 无验证集时 val_dice 恒为 -1，不能喂给调度器（会把 LR 一路降到下限）
                scheduler.step(val_dice)
                new_lr = optimizer.param_groups[0]["lr"]
                if new_lr < cur_lr:
                    note = (f"\n[学习率] 验证 Dice 连续 {args.lr_patience} 轮未提升："
                            f"{cur_lr:.2e} → {new_lr:.2e}")
                    cur_lr = new_lr
            extra = "".join(f" {n}_dice={d:.4f}"
                            for n, d in zip(config.CLASS_NAMES, per_ch_dice))
            log(f"[Epoch {epoch:03d}/{args.epochs}] loss={train_loss:.4f} "
                f"val_dice={val_dice:.4f} val_iou={val_iou:.4f} time={dt:.1f}s"
                f" lr={cur_lr:.2e}"
                + extra
                + (" *best*" if improved else "") + note)

            if bad_epochs >= args.patience and epoch >= 10:
                log(f"[提前停止] 连续 {args.patience} 轮验证 Dice 未提升，停止训练。")
                break
    except KeyboardInterrupt:
        log("[中断] 收到 Ctrl-C，保存已训练到当前轮的模型权重。")
        torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                    "val_dice": val_dice, "hparams": hparams,
                    "out_ch": N_CH, "norm": args.norm,
                    "class_names": list(config.CLASS_NAMES)},
                   ckpt_path)

    total = time.time() - t_start
    if best_epoch > 0:
        log(f"[完成] 最佳轮次: epoch {best_epoch} | 验证 Dice(根系) {best_val_dice:.4f} | "
            f"模型已保存: {ckpt_path}")
    else:
        log(f"[完成] 模型已保存: {ckpt_path}")
    log(f"[完成] 总训练用时 {total / 60:.1f} 分钟 | 日志: {log_path}")
    print(f"\n模型目录: {folder}\n日志文件: {log_path}")


def _pairs(data_dir):
    from common.dataset import discover_pairs
    return discover_pairs(data_dir)


if __name__ == "__main__":
    main()
