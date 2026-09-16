"""把推理结果 CSV（如 root_pictures.csv）画成折线图：**一个编号一张图**。

图的组织方式：
  - 每个**编号**（如 C001）单独一张折线图；
  - 图里每个**重复次序**（C001-1 / C001-2 / C001-3 / C001-4）是一条折线；
  - 横坐标 = 拍摄日期（20241229、20241231 …，按真实时间间隔排布，不是等距序号）；
  - 纵坐标 = 总根长 (px)。

输入 CSV 就是 inference.py 的产物（表头见下），文件名里带三样信息，本工具按它分组：

    root_C001-1_20241229CK.jpg
         └编号┘ └重复┘ └─日期─┘└后缀┘

所以**文件名格式必须保持** `…_{编号}-{重复}_{8位日期}{后缀}.{扩展名}`。
`check_ok` / `root_ok` 为「否」的行（检查范围异常 / 滞回被淹没）默认**跳过**，
避免把不可信的点画进趋势里。

**图片名里的空格会被去掉**（`plant_ S062-1_…` → `plant_S062-1_…`）：项目里带空格与不带
空格两种写法并存，本工具按项目约定统一先 `replace(" ", "")` 再解析。清理后的完整 CSV
会另存一份副本到输出目录（`{原名}_无空格.csv`），**原始文件不会被改动**。

用法（在项目根目录下运行）：
    python tool\\analysis_datas\\analysis_datas.py --csv "C:\\Users\\me\\Desktop\\root_pictures.csv"
    python tool\\analysis_datas\\analysis_datas.py --csv a.csv,b.csv        # 多个文件
    python tool\\analysis_datas\\analysis_datas.py                          # 用 analysis_datas/ 下所有 csv
    python tool\\analysis_datas\\analysis_datas.py --csv x.csv --ids C001,C203   # 只画这几个编号
    python tool\\analysis_datas\\analysis_datas.py --csv x.csv --no-overview     # 不生成总览图

输出：analysis_datas/{csv文件名}/{编号}.png（一张一个编号），外加
  - _总览.png       所有编号缩略图拼成一张，便于快速扫一遍哪个编号有异常
  - _汇总.csv       每个编号的日期数、各重复次序的点数、总根长范围，便于排查缺数据

配色用 dataviz 规范的前 4 个分类色槽，已通过色盲可辨性校验（见 readme）。
"""
import argparse
import csv
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates          # noqa: E402
import matplotlib.pyplot as plt            # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from common import naming  # noqa: E402

# ---- 文件名解析：…_{编号}-{重复}_{8位日期}{后缀}.{扩展名} ----
# 用 search 而不是 match：前缀可能是 `plant_ S062-1_...`（plant_ 后面带空格），
# 用 match 的话那个下划线会被前缀吃掉、只剩空格没法匹配，实测会漏掉 33 条。
NAME_PAT = re.compile(
    r"(?P<pid>[A-Za-z]+\d+)-(?P<rep>\d+)_(?P<date>\d{8})(?P<suffix>[A-Za-z]*)\.[A-Za-z]+$")

COL_NAME = "图片名"
COL_TOTAL = "总根长(px)"

# ---- 配色（dataviz 规范前 4 个分类色槽，已跑 validate_palette.js 校验通过）----
# 校验结果：明度带 PASS、彩度下限 PASS、色盲相邻对 ΔE 9.1 PASS、常视 ΔE 22.9 PASS；
# aqua(#1baf7a) 与 yellow(#eda100) 对底色对比度 <3:1 → 规范要求「可见标签兜底」，
# 所以本工具在图里同时给了图例 + 线端直接标注，不靠颜色单独承载身份。
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE = "#fcfcfb"     # 画布底色
INK = "#0b0b0b"         # 主文字
INK2 = "#52514e"        # 次文字
MUTED = "#898781"       # 坐标轴刻度
GRID = "#e1e0d9"        # 网格线（发丝级）
AXIS = "#c3c2b7"        # 轴线


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="把推理结果 CSV 按编号画成折线图（每图 4 条线 = 4 个重复次序）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=('示例：\n'
                '  python tool\\analysis_datas\\analysis_datas.py --csv "D:\\root_pictures.csv"\n'
                '  python tool\\analysis_datas\\analysis_datas.py --csv a.csv,b.csv --ids C001,C203\n'),
    )
    p.add_argument("--csv", default=None,
                   help="输入 CSV，逗号分隔可给多个；省略则用 analysis_datas/ 下所有 *.csv")
    p.add_argument("--out", type=Path, default=None,
                   help="输出根目录；默认 analysis_datas/（每个 csv 在其中各建一个同名文件夹）")
    p.add_argument("--ids", default=None,
                   help="只画这些编号，逗号分隔（如 C001,C203）；默认全画")
    p.add_argument("--dpi", type=int, default=150, help="单图分辨率，默认 150")
    p.add_argument("--no-overview", action="store_true", help="不生成 _总览.png")
    p.add_argument("--keep-bad", action="store_true",
                   help="保留 check_ok/root_ok=否 的点（默认跳过，避免不可信数据进趋势）")
    return p.parse_args(argv)


def load_csv(path: Path, keep_bad: bool):
    """读 CSV -> {(编号, 重复): {日期: 总根长}}，外加统计信息。

    返回 (data, stats)：stats 记录跳过的行数、无法解析文件名/数值的行数、重复点。
    """
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    rows = list(csv.DictReader(lines))
    fieldnames = list(rows[0].keys()) if rows else []
    data = defaultdict(dict)
    st = {"总行数": 0, "跳过不可信": 0, "文件名无法解析": 0, "数值无法解析": 0,
          "重复点": [], "含空格": 0}
    cleaned = []          # 清理过空格的完整行，用于另存一份副本
    for r in rows:
        st["总行数"] += 1
        raw = (r.get(COL_NAME) or "").strip()
        if raw.startswith("#"):
            continue                                   # 汇总注释行
        # 项目约定：文件名有 `plant_ S062-1_…`（带空格）与 `plant_S001-1_…` 两种写法并存，
        # 所有按名字解析的地方统一先 replace(" ", "")（见 readme「文件名不规范」那条）。
        name = raw.replace(" ", "")
        if name != raw:
            st["含空格"] += 1
        row = dict(r)
        row[COL_NAME] = name
        cleaned.append(row)
        m = NAME_PAT.search(name)
        if not m:
            st["文件名无法解析"] += 1
            continue
        ok = (r.get("check_ok", "是"), r.get("root_ok", "是"))
        if not keep_bad and ("否" in ok):
            st["跳过不可信"] += 1
            continue
        try:
            val = float(str(r.get(COL_TOTAL, "")).strip())
        except ValueError:
            st["数值无法解析"] += 1
            continue
        key = (m["pid"], m["rep"])
        if m["date"] in data[key]:
            st["重复点"].append(f"{name}（{m['date']}）")
        data[key][m["date"]] = val
    st["_cleaned"] = (fieldnames, cleaned)
    return data, st


def write_cleaned(csv_path: Path, fieldnames, cleaned, out_path: Path):
    """把去掉空格后的 CSV 另存一份副本（**不动原始文件**）。

    只改「图片名」一列，其余原样搬运，列顺序不变，可直接当原件用。
    """
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(cleaned)


def declutter(points, min_gap):
    """线端标签去重叠：按 y 排序后强制相邻间距 >= min_gap。

    points: [(y, 文本)]。返回 [(文本, 调整后的 y)]，按 y 升序 —— 只返回列表，
    不用 y 当字典键（两条线 y 相同时会互相覆盖）。
    """
    pts = sorted(points, key=lambda t: t[0])
    ys = [y for y, _ in pts]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < min_gap:
            ys[i] = ys[i - 1] + min_gap
    return [(txt, y) for (_, txt), y in zip(pts, ys)]


def plot_one(pid, series, out_path: Path, dpi: int):
    """series: [(重复, {日期: 总根长})]，已按重复次序排好。"""
    fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ends, lo, hi = [], float("inf"), float("-inf")
    for i, (rep, pts) in enumerate(series):
        color = SERIES[i % len(SERIES)]
        dates = sorted(pts)
        xs = [datetime.strptime(d, "%Y%m%d") for d in dates]
        ys = [pts[d] for d in dates]
        lo, hi = min(lo, min(ys)), max(hi, max(ys))
        ax.plot(xs, ys, color=color, lw=1.8, marker="o", ms=5.0,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3,
                label=f"{pid}-{rep}")
        ends.append((ys[-1], f"{pid}-{rep}"))

    # 纵轴从 0 起：根长是「量」，截断纵轴会把微小波动放大成陡坡
    span = max(hi - lo, 1.0)
    ax.set_ylim(0, hi + span * 0.28)

    # 线端直接标注（规范：≤4 条线要直接标注，身份不靠颜色单独承载）
    ylim = ax.get_ylim()
    min_gap = (ylim[1] - ylim[0]) * 0.062
    last_x = ax.get_xlim()[1]
    for txt, y1 in declutter(ends, min_gap):
        ax.annotate(txt, xy=(last_x, y1), xytext=(5, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=9, color=INK2, annotation_clip=False)

    ax.set_title(f"{pid} 总根长随时间变化", fontsize=14, color=INK,
                 loc="left", pad=12)
    ax.set_xlabel("拍摄日期", fontsize=10, color=INK2, labelpad=8)
    ax.set_ylabel("总根长 (px)", fontsize=10, color=INK2, labelpad=8)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y%m%d"))
    all_dates = sorted({d for _, pts in series for d in pts})
    if len(all_dates) <= 12:          # 点少就把每个日期都标出来，别让 AutoDateLocator 抽稀
        ticks = [datetime.strptime(d, "%Y%m%d") for d in all_dates]
        ax.set_xticks(ticks)
        ax.set_xticklabels([d for d in all_dates], rotation=45, ha="right")

    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    leg = ax.legend(frameon=False, fontsize=9, ncol=len(series),
                    loc="upper left", handlelength=1.6, columnspacing=1.4)
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.subplots_adjust(right=0.86, left=0.10, top=0.88, bottom=0.20)
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def plot_overview(pid_list, data, out_path: Path, dpi: int = 80):
    """所有编号的缩略图拼成一张总览，用来快速扫哪个编号有异常。"""
    n = len(pid_list)
    cols = min(20, max(1, int(n ** 0.5 * 1.6)))
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.15), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    axes = axes.ravel() if n > 1 else [axes]
    for ax, pid in zip(axes, pid_list):
        ax.set_facecolor(SURFACE)
        for i, (_, rep) in enumerate(sorted(r for r in data if r[0] == pid)):
            pts = data[(pid, rep)]
            ds = sorted(pts)
            # 带 marker：只有一个日期的编号（实测有，如 C0015）只画得出一根单点折线，
            # 没有 marker 的话整个格子是空的，会被误读成"这个编号没数据"
            ax.plot([datetime.strptime(d, "%Y%m%d") for d in ds],
                    [pts[d] for d in ds], color=SERIES[i % len(SERIES)],
                    lw=1.2, marker="o", ms=1.8)
        ax.set_title(pid, fontsize=6, color=INK2, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color(GRID)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("总览：每个编号一张缩略图（同色系含义同单图）", fontsize=10,
                 color=INK, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def main():
    args = parse_args()
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    out_root = args.out or (PROJECT_ROOT / "analysis_datas")
    if args.csv:
        csvs = [Path(s.strip()) for s in str(args.csv).split(",") if s.strip()]
    else:
        csvs = sorted(out_root.glob("*.csv"))
        if not csvs:
            sys.exit(f"[错误] 没给 --csv，{out_root} 下也没有 csv 文件")
    for c in csvs:
        if not c.is_file():
            sys.exit(f"[错误] 找不到 CSV: {c}")

    want = {s.strip() for s in str(args.ids).split(",")} if args.ids else None

    for csv_path in csvs:
        data, st = load_csv(csv_path, args.keep_bad)
        pids = sorted({k[0] for k in data})
        if want:
            pids = [p for p in pids if p in want]
        if not pids:
            print(f"[跳过] {csv_path.name}: 没有可画的编号\n")
            continue

        out_dir = naming.create_unique_dir(out_root, csv_path.stem)
        print(f"=== {csv_path.name} → {out_dir.name}/ ===")
        print(f"  共 {st['总行数']} 行 | 编号 {len(pids)} 个 | 跳过不可信 {st['跳过不可信']} 行"
              + (f" | 文件名无法解析 {st['文件名无法解析']} 行" if st["文件名无法解析"] else "")
              + (f" | 数值无法解析 {st['数值无法解析']} 行" if st["数值无法解析"] else ""))
        if st["含空格"]:
            fn, cl = st["_cleaned"]
            dst = out_dir / f"{csv_path.stem}_无空格.csv"
            write_cleaned(csv_path, fn, cl, dst)
            print(f"  [清理] {st['含空格']} 个图片名含空格，已按项目约定去掉"
                  f"（如 `plant_ S062-1_…` → `plant_S062-1_…`）；"
                  f"清理后的完整副本：{dst.name}（**原始文件未改动**）")
        if st["重复点"]:
            print(f"  [注意] {len(st['重复点'])} 个重复点（同一编号+重复+日期出现多次），"
                  f"已取最后一次：{st['重复点'][:3]}")

        summary = []
        for k, pid in enumerate(pids, 1):
            series = sorted((rep, pts) for (p, rep), pts in data.items() if p == pid)
            plot_one(pid, series, out_dir / f"{pid}.png", args.dpi)
            allv = [v for _, pts in series for v in pts.values()]
            summary.append([pid, len({d for _, pts in series for d in pts}),
                            ";".join(f"{rep}:{len(pts)}" for rep, pts in series),
                            f"{min(allv):.1f}", f"{max(allv):.1f}"])
            if k % 50 == 0 or k == len(pids):
                print(f"  已画 {k}/{len(pids)}")

        with open(out_dir / "_汇总.csv", "w", encoding="utf-8-sig", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["编号", "日期数", "各重复次序的点数", "总根长最小", "总根长最大"])
            wr.writerows(summary)
        if not args.no_overview:
            plot_overview(pids, data, out_dir / "_总览.png")
        print(f"  输出: {out_dir}\n")


if __name__ == "__main__":
    main()
