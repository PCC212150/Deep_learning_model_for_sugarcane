"""生成实验报告所需全部插图。

运行（项目根目录下）：
    C:\\Users\\21215\\.conda\\envs\\pcc\\python.exe experimental_report/make_figures.py
输出：experimental_report/figures/*.png
"""
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw
from skimage.morphology import disk, erosion, skeletonize
from skimage.measure import label as sk_label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)

import config  # noqa: E402
from common import gt_mask, image_io, predict  # noqa: E402
from common.rsml_parse import parse_rsml, root_stats  # noqa: E402
from common.skeleton_stats import analyze_mask_ex  # noqa: E402
from common.unet import UNet  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

MODEL_DIR = ROOT / "model" / "model_202609092044"
DISPLAY_W = 1300


def load_model():
    ckpt = torch.load(MODEL_DIR / f"{MODEL_DIR.name}.pth", map_location="cpu")
    model = UNet(3, 1)
    model.load_state_dict(ckpt["state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, device


def thumb(img, width=DISPLAY_W):
    h, w = img.shape[:2]
    return np.asarray(Image.fromarray(img).resize(
        (width, int(h * width / w)), Image.Resampling.LANCZOS))


def overlay(img, mask, alpha=0.45, color=(255, 0, 0)):
    out = img.astype(np.float32)
    if mask.any():
        tint = np.asarray(color, np.float32)
        out[mask] = out[mask] * (1 - alpha) + tint * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def up_prob(prob, h0, w0):
    """模型分辨率概率图 -> 原图分辨率 (h0, w0)。"""
    t = torch.from_numpy(prob)[None, None].float()
    up = torch.nn.functional.interpolate(t, size=(h0, w0), mode="bilinear",
                                         align_corners=False)
    return up[0, 0].numpy()


def fig1_dataset():
    """数据示例：原图 + RSML 标注叠加。"""
    name = "plant_ S062-1"
    img = image_io.load_rgb(ROOT / "datasets" / "test" / f"{name}.png")
    roots = parse_rsml(ROOT / "datasets" / "test" / f"{name}.rsml")
    mask = gt_mask.draw_mask_from_roots(roots, (img.shape[1], img.shape[0]),
                                        config.MASK_LINE_WIDTH)
    cnt, lens, total = root_stats(roots)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].imshow(thumb(img))
    axes[0].set_title("水培甘蔗根系原图（5472×3648）")
    axes[1].imshow(thumb(overlay(img, mask, alpha=0.55)))
    axes[1].set_title(f"RSML 标注叠加（红）：{cnt} 条根，总长 {total:.0f} px")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_dataset.png", bbox_inches="tight")
    plt.close(fig)


def fig2_pipeline():
    """整体技术路线流程图。"""
    steps = [
        ("数据准备", "datasets/train、test\npng + rsml 配对\n折线控制点解析"),
        ("真值掩码", "RSML 折线画 5px 线宽\n图像等比缩放至长边 1024/1536\n(16 的倍数)"),
        ("U-Net 训练", "编码-解码 + 跳跃连接\nBCE + Dice 损失\nAdam, 早停"),
        ("训练产物", "model_YYYYMMDDHHMM/\n.pth 权重 + 日志\n+ hparams.json"),
        ("推理", "整图缩放 → 概率图\n上采样回原图\n滞回阈值二值化"),
        ("骨架后处理", "骨架化 → 剪枝\n交叉点走向配对续接\n叶端计数 = 根数"),
        ("结果输出", "txt 统计（根数/逐根长/总长）\nmask.png / overlay.png\n预测 rsml 折线"),
    ]
    fig, ax = plt.subplots(figsize=(13.5, 3.4))
    ax.set_xlim(0, 7 * 1.9 + 0.2)
    ax.set_ylim(0, 1.6)
    ax.axis("off")
    for i, (title, body) in enumerate(steps):
        x = 0.15 + i * 1.9
        box = FancyBboxPatch((x, 0.35), 1.6, 1.0,
                             boxstyle="round,pad=0.06", linewidth=1.4,
                             edgecolor="#1f77b4", facecolor="#eaf2fb")
        ax.add_patch(box)
        ax.text(x + 0.8, 1.18, title, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color="#123c69")
        ax.text(x + 0.8, 0.75, body, ha="center", va="center", fontsize=8.2)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + 1.62, 0.85), (x + 1.9, 0.85),
                                         arrowstyle="-|>", mutation_scale=16,
                                         color="#333333", linewidth=1.6))
    fig.tight_layout()
    fig.savefig(FIG / "fig6_pipeline.png", bbox_inches="tight")
    plt.close(fig)


def fig3_unet():
    """U-Net 结构示意图。"""
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    enc = [("输入图像 3ch", 0.5, 4.4, "#d9d9d9"),
           ("64ch", 1.7, 3.6, "#c6dbef"),
           ("128ch", 2.9, 2.9, "#9ecae1"),
           ("256ch", 4.1, 2.2, "#6baed6"),
           ("512ch", 5.3, 1.5, "#3182bd")]
    for text, x, y, color in enc:
        h = 0.5 + y * 0.35
        ax.add_patch(FancyBboxPatch((x - 0.45, y - h / 2), 0.95, h,
                                    boxstyle="round,pad=0.03",
                                    facecolor=color, edgecolor="#444"))
        ax.text(x, y, text, ha="center", va="center", fontsize=9)
    ax.add_patch(FancyBboxPatch((6.5 - 0.5, 0.55), 1.0, 0.85,
                                boxstyle="round,pad=0.03",
                                facecolor="#08519c", edgecolor="#444"))
    ax.text(6.5, 0.97, "瓶颈 1024ch", ha="center", va="center", fontsize=9,
            color="white")
    up = [("512ch", 7.6, 1.5, "#3182bd"), ("256ch", 8.8, 2.2, "#6baed6"),
          ("128ch", 10.0, 2.9, "#9ecae1"), ("64ch", 11.2, 3.6, "#c6dbef")]
    for text, x, y, color in up:
        h = 0.5 + y * 0.35
        ax.add_patch(FancyBboxPatch((x - 0.45, y - h / 2), 0.95, h,
                                    boxstyle="round,pad=0.03",
                                    facecolor=color, edgecolor="#444"))
        ax.text(x, y, text, ha="center", va="center", fontsize=9)
    ax.add_patch(FancyBboxPatch((11.9, 4.4), 0.9, 0.7,
                                boxstyle="round,pad=0.03",
                                facecolor="#fdd0a2", edgecolor="#444"))
    ax.text(12.35, 4.75, "输出 1ch", ha="center", va="center", fontsize=9)
    for (_, x1, y1, _), (_, x2, y2, _) in zip(enc[:-1], enc[1:]):
        ax.add_patch(FancyArrowPatch((x1 + 0.5, y1 - 0.15), (x2 - 0.5, y2 + 0.15),
                                     arrowstyle="-|>", mutation_scale=13,
                                     color="#666", linewidth=1.2))
    ax.add_patch(FancyArrowPatch((5.3 + 0.5, 1.2), (6.5 - 0.5, 1.0),
                                 arrowstyle="-|>", mutation_scale=13,
                                 color="#666", linewidth=1.2))
    ax.add_patch(FancyArrowPatch((6.5 + 0.5, 1.0), (7.6 - 0.5, 1.2),
                                 arrowstyle="-|>", mutation_scale=13,
                                 color="#666", linewidth=1.2))
    for (_, x1, y1, _), (_, x2, y2, _) in zip(reversed(up[:-1]), reversed(up[1:])):
        ax.add_patch(FancyArrowPatch((x1 + 0.5, y1 - 0.15), (x2 - 0.5, y2 + 0.15),
                                     arrowstyle="-|>", mutation_scale=13,
                                     color="#666", linewidth=1.2))
    ax.add_patch(FancyArrowPatch((11.2 + 0.5, 3.7), (11.9, 4.5),
                                 arrowstyle="-|>", mutation_scale=13,
                                 color="#666", linewidth=1.2))
    # 跳跃连接：从各编码块顶部绕上方连到对应解码块顶部
    def top_y(y):
        return y + (0.5 + y * 0.35) / 2
    skip = [(1.7, 3.6, 11.2, 3.6, -0.30), (2.9, 2.9, 10.0, 2.9, -0.34),
            (4.1, 2.2, 8.8, 2.2, -0.40), (5.3, 1.5, 7.6, 1.5, -0.46)]
    for x1, y1, x2, y2, rad in skip:
        ax.add_patch(FancyArrowPatch((x1, top_y(y1) + 0.05), (x2, top_y(y2) + 0.05),
                                     arrowstyle="-|>", mutation_scale=12,
                                     color="#d62728", linewidth=1.3,
                                     linestyle="--",
                                     connectionstyle=f"arc3,rad={rad}"))
    ax.text(6.5, 6.15, "U-Net 结构：4 次下采样编码 + 瓶颈 + 4 次转置卷积上采样解码",
            ha="center", fontsize=12.5, fontweight="bold")
    ax.text(6.5, 5.75, "红色虚线 = 跳跃连接（池化前的编码特征直接拼接给解码器，保留细根细节）",
            ha="center", fontsize=9.5, color="#d62728")
    ax.text(3.6, 0.35, "每次下采样：2×2 最大池化", fontsize=8.5, color="#555")
    ax.text(9.6, 0.35, "每次上采样：ConvTranspose2d(k=2,s=2)", fontsize=8.5,
            color="#555")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_unet.png", bbox_inches="tight")
    plt.close(fig)


def fig4_training():
    """训练曲线（来自训练日志）。"""
    log = MODEL_DIR / f"{MODEL_DIR.name}_log.txt"
    epochs, losses, dice, ious = [], [], [], []
    pat = re.compile(r"\[Epoch (\d+)/\d+\] loss=([\d.]+) val_dice=([\d.]+) "
                     r"val_iou=([\d.]+)")
    for line in log.read_text(encoding="utf-8").splitlines():
        m = pat.search(line)
        if m:
            epochs.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            dice.append(float(m.group(3)))
            ious.append(float(m.group(4)))
    best_e = epochs[int(np.argmax(dice))]
    best_d = max(dice)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    axes[0].plot(epochs, losses, color="#1f77b4", linewidth=1.4)
    axes[0].set_title("训练损失（BCE + Dice）")
    axes[0].set_xlabel("训练轮次 epoch")
    axes[0].set_ylabel("损失")
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, dice, color="#2ca02c", linewidth=1.4, label="验证 Dice")
    axes[1].plot(epochs, ious, color="#ff7f0e", linewidth=1.2, label="验证 IoU")
    axes[1].axvline(best_e, color="#d62728", linestyle="--", linewidth=1.1)
    axes[1].annotate(f"最佳轮次 {best_e}\nDice={best_d:.3f}",
                     xy=(best_e, best_d), xytext=(best_e + 12, best_d - 0.12),
                     arrowprops=dict(arrowstyle="->", color="#d62728"),
                     fontsize=10, color="#d62728")
    axes[1].set_title("验证集指标（每轮）")
    axes[1].set_xlabel("训练轮次 epoch")
    axes[1].set_ylabel("Dice / IoU")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(bottom=0)
    fig.suptitle(f"训练曲线（共 {len(epochs)} 轮，验证 Dice 连续 60 轮未提升触发早停；"
                 f"RTX 3090 用时 13.2 分钟）", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_training.png", bbox_inches="tight")
    plt.close(fig)
    return dict(epochs=len(epochs), best_epoch=best_e, best_dice=best_d)


def fig5_prediction(model, device):
    """预测结果四联图：原图 / 标注 / 概率图 / 预测叠加。"""
    name = "plant_ S062-1"
    img = image_io.load_rgb(ROOT / "datasets" / "test" / f"{name}.png")
    roots = parse_rsml(ROOT / "datasets" / "test" / f"{name}.rsml")
    gmask = gt_mask.draw_mask_from_roots(roots, (img.shape[1], img.shape[0]),
                                         config.MASK_LINE_WIDTH)
    res = predict.predict(model, img, max_side=config.MAX_SIDE,
                          stride=config.STRIDE, device=device,
                          low_thresh=config.PRED_LOW_THRESHOLD)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].imshow(thumb(img))
    axes[0, 0].set_title("(a) 原图")
    axes[0, 1].imshow(thumb(overlay(img, gmask, alpha=0.55)))
    axes[0, 1].set_title("(b) RSML 真值标注")
    prob = res["prob_target"]
    im = axes[1, 0].imshow(prob, cmap="inferno")
    axes[1, 0].set_title(f"(c) 模型输出的根系概率图（{prob.shape[1]}×{prob.shape[0]}）")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.035)
    axes[1, 1].imshow(thumb(overlay(img, res["mask_orig"], alpha=0.5)))
    st = analyze_mask_ex(res["mask_orig"], spur=config.PRED_SPUR_LENGTH,
                         min_len=config.MIN_ROOT_LENGTH)
    axes[1, 1].set_title(f"(d) 预测结果叠加：{st['count']} 条根，"
                         f"总长 {st['total']:.0f} px")
    for ax in axes.ravel():
        ax.axis("off") if ax is not axes[1, 0] else None
    axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])
    fig.tight_layout()
    fig.savefig(FIG / "fig7_prediction.png", bbox_inches="tight")
    plt.close(fig)


def fig6_skeleton(model, device):
    """骨架统计原理：骨架 + 叶端/分叉点。"""
    name = "plant_ S062-1"
    img = image_io.load_rgb(ROOT / "datasets" / "test" / f"{name}.png")
    res = predict.predict(model, img, max_side=config.MAX_SIDE,
                          stride=config.STRIDE, device=device,
                          low_thresh=config.PRED_LOW_THRESHOLD)
    m = erosion(res["mask_orig"], footprint=disk(1))
    skel = skeletonize(m)
    ys, xs = np.nonzero(skel)
    pts = {(int(y), int(x)) for y, x in zip(ys.tolist(), xs.tolist())}
    leaves, juncs = [], []
    for (y, x) in pts:
        d = sum((y + dy, x + dx) in pts for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if not (dx == 0 and dy == 0))
        if d == 1:
            leaves.append((y, x))
        elif d >= 3:
            juncs.append((y, x))
    vis = img.copy()
    vis[skel] = [255, 255, 0]
    pv = Image.fromarray(vis)
    d = ImageDraw.Draw(pv)
    for (y, x) in juncs:
        d.ellipse([x - 10, y - 10, x + 10, y + 10], outline=(255, 60, 60), width=4)
    for (y, x) in leaves:
        d.ellipse([x - 15, y - 15, x + 15, y + 15], outline=(0, 220, 0), width=4)
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    ax.imshow(thumb(np.asarray(pv)))
    ax.axis("off")
    ax.set_title(f"预测掩码骨架化示意（剪枝前的原始骨架）：黄线=骨架中心线，"
                 f"绿圈=叶端（{len(leaves)} 个），红圈=分叉/交叉点（{len(juncs)} 个）\n"
                 f"统计时还需：剪枝去毛刺 → 交叉点按走向配对续接 → 每条轨迹=1 根",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_skeleton.png", bbox_inches="tight")
    plt.close(fig)


def fig7_and_fig8(model, device):
    """16 张训练图的计数/总长对比 + (滞回低阈值 × 剪枝阈值) 网格搜索。"""
    names = sorted(p.stem for p in (ROOT / "datasets" / "train").glob("*.png"))
    gt_cnt, gt_tot = {}, {}
    for n in names:
        c, _, t = root_stats(parse_rsml(
            ROOT / "datasets" / "train" / f"{n}.rsml"))
        gt_cnt[n], gt_tot[n] = c, t

    los = [0.10, 0.15, 0.20, 0.30]
    spurs = [40, 60, 80, 100, 120]
    mae = np.zeros((len(los), len(spurs)))
    pred_cnt, pred_tot = {}, {}
    for n in names:
        img = image_io.load_rgb(ROOT / "datasets" / "train" / f"{n}.png")
        res = predict.predict(model, img, max_side=config.MAX_SIDE,
                              stride=config.STRIDE, device=device,
                              low_thresh=0)
        h0, w0 = img.shape[:2]
        up = up_prob(res["prob_target"], h0, w0)
        weak = up > min(los)
        lab = sk_label(weak, connectivity=2)
        strong = up > 0.5
        for i, lo in enumerate(los):
            weak_i = up > lo
            lab_i = sk_label(weak_i, connectivity=2)
            keep = np.unique(lab_i[strong]); keep = keep[keep > 0]
            m = np.isin(lab_i, keep)
            for j, spur in enumerate(spurs):
                st = analyze_mask_ex(m, spur=spur,
                                     min_len=config.MIN_ROOT_LENGTH)
                mae[i, j] += abs(st["count"] - gt_cnt[n]) / len(names)
                if lo == 0.10 and spur == 60:
                    pred_cnt[n], pred_tot[n] = st["count"], st["total"]

    # ---- fig7: 计数柱状对比 + 总长散点 ----
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    axes[0].bar(x - 0.2, [gt_cnt[n] for n in names], width=0.4,
                label="RSML 真值", color="#4c72b0")
    axes[0].bar(x + 0.2, [pred_cnt[n] for n in names], width=0.4,
                label="预测", color="#dd8452")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([n.replace("plant_", "").replace("plant", "")
                             for n in names], rotation=45, ha="right",
                            fontsize=7.5)
    axes[0].legend()
    axes[0].set_ylabel("根数量")
    mae_cnt = np.mean([abs(pred_cnt[n] - gt_cnt[n]) for n in names])
    axes[0].set_title(f"每张图根数量：真值 vs 预测（平均绝对误差 {mae_cnt:.1f} 根）")
    axes[0].grid(axis="y", alpha=0.3)
    g = [gt_tot[n] for n in names]
    p = [pred_tot[n] for n in names]
    axes[1].scatter(g, p, color="#c44e52", zorder=3)
    lim = [0, max(g + p) * 1.1]
    axes[1].plot(lim, lim, "--", color="#555", linewidth=1, label="y = x")
    axes[1].set_xlim(lim); axes[1].set_ylim(lim)
    axes[1].set_xlabel("真值总长 (px)")
    axes[1].set_ylabel("预测总长 (px)")
    ratio = np.mean([pred_tot[n] / gt_tot[n] for n in names])
    axes[1].set_title(f"每张图根系总长：预测/真值平均 {ratio:.2f}")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig8_count_total.png", bbox_inches="tight")
    plt.close(fig)

    # ---- fig8: 网格搜索热图 ----
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    im = ax.imshow(mae, cmap="viridis_r")
    ax.set_xticks(range(len(spurs)), [str(s) for s in spurs])
    ax.set_yticks(range(len(los)), [f"{l:.2f}" for l in los])
    ax.set_xlabel("骨架剪枝阈值 spur (px)")
    ax.set_ylabel("滞回低阈值 lo")
    ax.set_title("根数平均绝对误差（16 张训练图网格扫描）")
    for i in range(len(los)):
        for j in range(len(spurs)):
            ax.text(j, i, f"{mae[i, j]:.1f}", ha="center", va="center",
                    color="white" if mae[i, j] > mae.min() * 1.2 else "black",
                    fontsize=9)
    bi, bj = np.unravel_index(np.argmin(mae), mae.shape)
    ax.scatter([bj], [bi], marker="s", s=220, facecolor="none",
               edgecolor="red", linewidth=2)
    ax.set_title(f"根数平均绝对误差（16 张训练图）\n"
                 f"最优：lo={los[bi]:.2f}, spur={spurs[bj]}px → MAE={mae.min():.2f} 根",
                 fontsize=10.5)
    fig.colorbar(im, ax=ax, fraction=0.045, label="MAE（根）")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_grid_search.png", bbox_inches="tight")
    plt.close(fig)
    return dict(mae_best=float(mae.min()), mae_cnt=float(mae_cnt),
                ratio_total=float(ratio),
                n_gt=int(sum(gt_cnt.values())),
                n_pred=int(sum(pred_cnt.values())))


def main():
    want = set(sys.argv[1:])  # 可选：只生成指定图，如 `... make_figures.py fig3`
    need_model = (not want) or bool(want & {"fig5", "fig6", "fig7", "fig8"})
    model, device = load_model() if need_model else (None, None)
    if not want or "fig1" in want:
        fig1_dataset()
    if not want or "fig2" in want:
        fig2_pipeline()
    if not want or "fig3" in want:
        fig3_unet()
    if not want or "fig4" in want:
        info = fig4_training()
        print("训练: 轮数=%d, 最佳轮=%d, 最佳val_dice=%.4f" % (
            info["epochs"], info["best_epoch"], info["best_dice"]))
    if not want or "fig5" in want:
        fig5_prediction(model, device)
    if not want or "fig6" in want:
        fig6_skeleton(model, device)
    if not want or (want & {"fig7", "fig8"}):
        stats = fig7_and_fig8(model, device)
        print("统计: " + str(stats))
    print("图已输出到", FIG)


if __name__ == "__main__":
    main()
