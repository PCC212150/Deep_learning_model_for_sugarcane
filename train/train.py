"""训练 U-Net 分割甘蔗根系。

运行（pcc 环境，建议在项目根目录）：
    conda activate pcc
    python train/train.py                        # 默认 200 轮 / 早停 60
    python train/train.py --size 1536 --batch 8  # 部署到 4xRTX3090 服务器时

每轮输出：轮次 / 损失 / 验证Dice / 本轮耗时(s)（控制台 + 模型文件夹内日志）。
模型保存：model/model_YYYYMMDDHHMM/（验证最优轮权重 .pth + 参数日志 .txt + hparams.json）
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
from common.dataset import RootDataset  # noqa: E402
from common.unet import UNet  # noqa: E402


def dice_loss(prob, target):
    """prob: sigmoid 后概率, target: 0/1。返回 batch 平均 Dice 损失。"""
    eps = 1.0
    inter = (prob * target).sum(dim=(1, 2, 3))
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + eps) / (den + eps)).mean()


def parse_args():
    p = argparse.ArgumentParser(description="训练甘蔗根系 U-Net")
    p.add_argument("--size", type=int, default=config.MAX_SIDE, help="输入长边像素")
    p.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--patience", type=int, default=config.PATIENCE)
    p.add_argument("--val-size", type=int, default=config.VAL_SIZE,
                   help="从训练数据抽出的验证组数")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--data-dir", type=Path, default=config.TRAIN_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=config.MODEL_DIR)
    p.add_argument("--no-amp", action="store_true", help="关闭混合精度")
    p.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    return p.parse_args()


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

    # ---- 数据划分：固定随机抽取验证组（保证可复现） ----
    names = [n for n, _, _ in _pairs(args.data_dir)]
    rng = np.random.RandomState(args.seed)
    rng.shuffle(names)
    val_names = sorted(names[:args.val_size])
    train_names = sorted(names[args.val_size:])
    print(f"数据: 共 {len(names)} 组 | 训练 {len(train_names)} 组 | "
          f"验证 {len(val_names)} 组")
    print(f"验证组: {', '.join(val_names) if val_names else '无(不早停,保存最后轮)'}")
    t0 = time.time()
    train_ds = RootDataset(args.data_dir, names=train_names,
                           max_side=args.size, augment=True, seed=args.seed)
    val_ds = RootDataset(args.data_dir, names=val_names,
                         max_side=args.size, augment=False, seed=args.seed)
    print(f"数据加载完成，用时 {time.time() - t0:.1f}s")

    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch,
                                         shuffle=True, drop_last=True,
                                         num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch,
                                             shuffle=False, num_workers=0)

    # ---- 模型 ----
    model = UNet(in_ch=3, out_ch=1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net 参数量: {n_params / 1e6:.2f}M | 输入长边 {args.size} | "
          f"batch {args.batch}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.02)
    amp = (device.type == "cuda") and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = torch.nn.BCEWithLogitsLoss()

    # ---- 输出目录：model/model_年月日时分，重名追加 -1/-2… ----
    ts = naming.timestamp()
    folder = naming.unique_path(args.out_dir / naming.model_folder_name(ts))
    folder.mkdir(parents=True, exist_ok=False)
    ckpt_path = folder / f"{folder.name}.pth"
    log_path = folder / f"{folder.name}_log.txt"
    hparams = {k: (str(v) if isinstance(v, Path) else v)
               for k, v in vars(args).items()}
    hparams.update({"device": str(device), "gpu": torch.cuda.get_device_name(0)
                    if device.type == "cuda" else "cpu",
                    "val_names": val_names, "train_names": train_names,
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
        f"输入 {args.size} | batch {args.batch} | lr {args.lr} | "
        f"epochs {args.epochs} | 训练 {len(train_names)} 组 | 验证 {len(val_names)} 组")
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
            for x, y, _ in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=amp):
                    out = model(x)
                    prob = torch.sigmoid(out)
                    loss = criterion(out, y) + dice_loss(prob, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.item())
                n_batch += 1
            train_loss = loss_sum / max(n_batch, 1)

            # ---- 验证（轮内） ----
            val_dice = val_iou = -1.0
            if val_ds:
                model.eval()
                d_list, i_list = [], []
                with torch.no_grad():
                    for x, y, _ in val_loader:
                        x = x.to(device)
                        with torch.autocast(device_type="cuda", enabled=amp):
                            prob = torch.sigmoid(model(x)).float()
                        pb = prob.cpu().numpy() > 0.5
                        gt = y.numpy() > 0.5
                        for i in range(len(gt)):
                            tp = (pb[i] & gt[i]).sum()
                            d = 2.0 * tp / (pb[i].sum() + gt[i].sum() + 1e-8)
                            iou = tp / (pb[i].sum() + gt[i].sum() - tp + 1e-8)
                            d_list.append(float(d))
                            i_list.append(float(iou))
                val_dice = float(np.mean(d_list)) if d_list else -1.0
                val_iou = float(np.mean(i_list)) if i_list else -1.0

            improved = val_dice - best_val_dice > 1e-4
            if improved:
                best_val_dice, best_epoch = val_dice, epoch
                torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                            "val_dice": val_dice, "hparams": hparams},
                           ckpt_path)
                bad_epochs = 0
            else:
                bad_epochs += 1

            dt = time.time() - t_ep
            scheduler.step()
            log(f"[Epoch {epoch:03d}/{args.epochs}] loss={train_loss:.4f} "
                f"val_dice={val_dice:.4f} val_iou={val_iou:.4f} "
                f"time={dt:.1f}s"
                + (" *best*" if improved else ""))

            if bad_epochs >= args.patience and epoch >= 10:
                log(f"[提前停止] 连续 {args.patience} 轮验证 Dice 未提升，停止训练。")
                break
    except KeyboardInterrupt:
        log("[中断] 收到 Ctrl-C，保存已训练到当前轮的模型权重。")
        torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                    "val_dice": val_dice, "hparams": hparams}, ckpt_path)

    total = time.time() - t_start
    if best_epoch > 0:
        log(f"[完成] 最佳轮次: epoch {best_epoch} | 验证 Dice {best_val_dice:.4f} | "
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
