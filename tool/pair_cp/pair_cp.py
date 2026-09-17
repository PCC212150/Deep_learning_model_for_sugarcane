"""为实现 **C/P 对照**，把无法成对的图片从两边的文件夹里剔除。

C（`root_C###-#_<日期>CK.jpg`）与 P（`root_P###-#_<日期>PEG.jpg`）是两个处理，
做对照实验要求**同一个「编号-重复-日期」两边各有一张**。实际数据里有三种坑：

1. **异常命名**：不符合标准命名的文件（如 `root_P137-1 (2)_20250108PEG.jpg`）——
   归属本身就不确定，留着会污染对照；
2. **单边缺某天**：同一天只有 C 有、P 没有（或反过来），凑不成一对；
3. 上面两种都会让两边的数目对不上。

规则（**纯按文件名**，不改名、不按图像内容猜归属）：

    异常命名的           -> 剔除
    两边都有的日期        -> 保留
    只有一边有的日期      -> **两边都剔除**（保证严格一一对应）

剔除的文件**移到隔离目录**（默认 `<dir>\\_剔除_CP对照\\<C|P>\\`）而不是直接删，
配合日志可以随时搬回来。确认无误后自行删掉隔离目录即可。

用法：
    python pair_cp.py --dir "E:\\baiduwangpan\\DownLoad" --dry-run   # 先预览
    python pair_cp.py --dir "E:\\baiduwangpan\\DownLoad"             # 正式执行
    python pair_cp.py --dir "..." --quarantine "D:\\回收"             # 指定隔离目录

执行后在 `--dir` 下生成 `CP对照_剔除日志.csv`（UTF-8 BOM，Excel 直接打开），
逐条记录「哪一侧 / 哪个文件 / 编号-重复 / 日期 / 剔除原因 / 搬到哪了」。
"""
import argparse
import csv
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# 标准命名：root_<C|P><编号>-<重复>_<8位日期><批次后缀>.jpg
STRICT = re.compile(r"^root_([CP])(\d+)-(\d+)_(\d{8})([A-Za-z]*)\.jpg$")

DEFAULT_SIDES = "C,P"
QUARANTINE_NAME = "_剔除_CP对照"
LOG_NAME = "CP对照_剔除日志.csv"

REASON_BAD = "异常命名"
REASON_ONLY = "{side} 独有（对侧缺该日）"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="剔除 C/P 无法成对的图片（异常命名的 + 单边独有的）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("示例：\n"
                '  python pair_cp.py --dir "E:\\baiduwangpan\\DownLoad" --dry-run\n'
                '  python pair_cp.py --dir "E:\\baiduwangpan\\DownLoad"\n'),
    )
    p.add_argument("--dir", required=True, help="含 C 与 P 两个子文件夹的父目录")
    p.add_argument("--sides", default=DEFAULT_SIDES,
                   help=f"两侧的文件夹名，逗号分隔（默认 {DEFAULT_SIDES}）")
    p.add_argument("--quarantine", default=None,
                   help=f"剔除文件的去处（默认 <dir>\\{QUARANTINE_NAME}）")
    p.add_argument("--dry-run", action="store_true",
                   help="只报告要剔除哪些，不移动任何文件、不写日志")
    return p.parse_args(argv)


def parse_side(root: Path, kind: str):
    """返回 ({(编号,重复,日期): 路径}, [异常命名的文件])。"""
    ok, bad = {}, []
    for f in sorted(root.iterdir()):
        if not f.is_file():
            continue
        m = STRICT.match(f.name)
        if m and m.group(1) == kind:
            key = (m.group(2), m.group(3), m.group(4))
            if key in ok:
                bad.append(f)               # 同一天重复出现，同样交给人看
            else:
                ok[key] = f
        else:
            bad.append(f)
    return ok, bad


def main(argv=None):
    args = parse_args(argv)
    base = Path(args.dir)
    if not base.is_dir():
        sys.exit(f"[错误] 文件夹不存在: {base}")
    sides = [s.strip() for s in args.sides.split(",") if s.strip()]
    if len(sides) != 2:
        sys.exit(f"[错误] --sides 需要正好两个文件夹名，当前: {args.sides}")

    data = {}
    for s in sides:
        d = base / s
        if not d.is_dir():
            sys.exit(f"[错误] 子文件夹不存在: {d}")
        ok, bad = parse_side(d, s)
        data[s] = (ok, bad)
        print(f"{s}: 标准命名 {len(ok)} 个 / 异常命名 {len(bad)} 个 / 共 {len(ok)+len(bad)} 个")

    a, b = sides
    ka, kb = set(data[a][0]), set(data[b][0])
    keep = ka & kb
    only_a, only_b = ka - kb, kb - ka
    print(f"\n两边都有（可对照）: {len(keep)} 个「编号-重复-日期」")
    print(f"{a} 独有（{b} 缺该日）: {len(only_a)} 个")
    print(f"{b} 独有（{a} 缺该日）: {len(only_b)} 个")

    moves = []                                   # (路径, 侧, 原因, 编号-重复, 日期)
    for s in sides:
        for f in data[s][1]:
            moves.append((f, s, REASON_BAD, "", ""))
    for key in sorted(only_a):
        m = STRICT.match(data[a][0][key].name)
        moves.append((data[a][0][key], a, REASON_ONLY.format(side=a),
                      f"{m.group(2)}-{m.group(3)}", m.group(4)))
    for key in sorted(only_b):
        m = STRICT.match(data[b][0][key].name)
        moves.append((data[b][0][key], b, REASON_ONLY.format(side=b),
                      f"{m.group(2)}-{m.group(3)}", m.group(4)))

    by_reason = defaultdict(list)
    for mv in moves:
        by_reason[mv[2]].append(mv)
    print("\n=== 将剔除 ===")
    for reason, items in by_reason.items():
        print(f"  [{reason}] {len(items)} 个")
        for f, s, _, _, _ in items[:5]:
            print(f"      {s}/{f.name}")
        if len(items) > 5:
            print(f"      … 其余 {len(items)-5} 个（详见日志）")
    for s in sides:
        n_after = len(data[s][0]) - sum(1 for mv in moves if mv[1] == s)
        print(f"  {s} 剩余 {n_after} 个")

    if args.dry_run:
        print("\n（预览模式，未移动任何文件，也没写日志）")
        return 0

    q = Path(args.quarantine) if args.quarantine else base / QUARANTINE_NAME
    for f, s, _, _, _ in moves:
        dst = q / s / f.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(dst))

    log = base / LOG_NAME
    with open(log, "w", encoding="utf-8-sig", newline="") as fh:
        fh.write(f"# C/P 对照裁剪日志    父目录: {base}\n")
        fh.write(f"# 保留 {len(keep)} 组「编号-重复-日期」（两边各一张，严格一一对应）；"
                 f"剔除 {len(moves)} 个文件\n")
        fh.write(f"# 规则：异常命名先剔除；再保留两边都有的日期；只有一边有的，两边都剔除\n")
        fh.write(f"# 剔除的文件在隔离目录里（原样未改），位置见下表最后一列\n")
        fh.write(f"# 隔离目录: {q}\n")
        w = csv.writer(fh)
        w.writerow(["侧", "文件名", "编号-重复", "日期", "剔除原因", "隔离位置"])
        for f, s, reason, key, date in sorted(moves, key=lambda mv: (mv[2], mv[1], mv[0].name)):
            w.writerow([s, f.name, key, date, reason, str(q / s / f.name)])
    print(f"\n已剔除 {len(moves)} 个文件 -> {q}")
    print(f"日志: {log}")

    # 复核：裁完两边必须严格一一对应
    for s in sides:
        ok, bad = parse_side(base / s, s)
        data[s] = (ok, bad)
        if bad:
            print(f"  [警告] {s} 里仍有 {len(bad)} 个异常文件: {[f.name for f in bad[:3]]}")
    fa, fb = set(data[a][0]), set(data[b][0])
    print(f"\n复核: {a} {len(data[a][0])} 个 / {b} {len(data[b][0])} 个；"
          f"差集 {len(fa - fb)} / {len(fb - fa)}  "
          + ("**严格一一对应**" if fa == fb else "**仍未对齐！**"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
