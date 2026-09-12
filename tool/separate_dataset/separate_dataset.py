"""把数据集文件夹按比例划分为 train / test / val 三份（验证集默认不分配）。

划分单位是「组」而不是单个文件：源文件夹里同名的一批文件算一组
（如 plant_ S062-1.png 与 plant_ S062-1.rsml），整组进同一份，不会把图片和标注拆散
（对应 common/dataset.py 的 discover_pairs：图片必须与同名 .rsml 同目录成对）。

用法：
    python separate_dataset.py --dir "C:\\Users\\21215\\Desktop\\RootTracer_RSML\\20251116ST"
    python separate_dataset.py --dir "D:\\数据\\20251116ST" --test 0.2
    python separate_dataset.py --dir "D:\\数据\\20251116ST" --train 0.7 --test 0.2 --val 0.1

输出：默认写到 <源文件夹同级>/<源文件夹名>_split/{train,test,val}/（重名自动加 -1），
      并生成 split.txt 记录本次划分的比例、种子和每组归属，便于复现。
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from common import naming  # noqa: E402

SUBSETS = ("train", "test", "val")

# 默认比例：训练 0.8 / 测试 0.2 / 验证 0（验证集默认不分配）
DEFAULT_TRAIN = 0.8
DEFAULT_TEST = 0.2

# 顺手的垃圾文件，不参与划分
JUNK_FILES = {".ds_store", "thumbs.db", "desktop.ini"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="把数据集文件夹按比例划分为 train / test / val（同名文件整组划分，配对不会拆散）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            '  python separate_dataset.py --dir "C:\\Users\\21215\\Desktop\\RootTracer_RSML\\20251116ST"\n'
            "      -> 0.8 训练 / 0.2 测试，不分配验证集\n"
            '  python separate_dataset.py --dir "D:\\数据\\20251116ST" --train 0.7 --test 0.2 --val 0.1\n'
            "\n"
            "只写 --test（或只写 --train）时，另一项按剩余自动推算；"
            "两者都写则必须和为 1。\n"
            "同一种子 + 同一份数据 = 同样的划分结果。"
        ),
    )
    parser.add_argument("--dir", required=True, help="源数据集文件夹")
    parser.add_argument("--out", default=None,
                        help="输出目录，默认 <源文件夹同级>/<源文件夹名>_split")
    parser.add_argument("--train", type=float, default=None, help=f"训练集比例，默认 {DEFAULT_TRAIN}")
    parser.add_argument("--test", type=float, default=None, help=f"测试集比例，默认 {DEFAULT_TEST}")
    parser.add_argument("--val", type=float, default=None, help="验证集比例，默认 0（不分配）")
    parser.add_argument("--seed", type=int, default=config.SEED, help=f"随机种子，默认 {config.SEED}")
    parser.add_argument("--move", action="store_true", help="移动文件（默认复制，源数据保留）")
    parser.add_argument("--dry-run", action="store_true", help="只预览划分结果，不写任何文件")
    return parser.parse_args(argv)


def resolve_ratios(args):
    """把 --train/--test/--val 解析成和为 1 的三份比例，返回 (ratios, 错误信息)。

    --train 与 --test 可以只写其中一个，另一个按剩余自动推算，例如：
        --test 0.2            -> train 0.8 / test 0.2 / val 0
        --train 0.7 --val 0.1 -> train 0.7 / test 0.2 / val 0.1
    两个都写时必须是显式自洽的（之和为 1），写错就直接报错而不是悄悄改数。
    """
    for name, value in (("train", args.train), ("test", args.test), ("val", args.val)):
        if value is not None and value < 0:
            return None, f"--{name} 不能为负数（当前 {value:g}）"

    val = 0.0 if args.val is None else args.val
    if args.train is not None and args.test is not None:
        train, test = args.train, args.test
        total = train + test + val
        if abs(total - 1.0) > 1e-6:
            return None, (f"比例之和必须为 1，当前为 {total:g}"
                          f"（train {train:g} + test {test:g} + val {val:g}）；"
                          f"只想指定其中一两项时，留空的那项会自动推算")
    elif args.train is not None:
        train, test = args.train, 1.0 - args.train - val
    elif args.test is not None:
        test, train = args.test, 1.0 - args.test - val
    else:
        test, train = DEFAULT_TEST, 1.0 - DEFAULT_TEST - val

    if train < -1e-9 or test < -1e-9:
        return None, (f"比例超出 1：train {train:g} / test {test:g} / val {val:g}，请调小 --val 或其他比例")
    return {"train": max(train, 0.0), "test": max(test, 0.0), "val": val}, None


def collect_groups(src):
    """按文件名主干分组：一组 = 同名的一批文件（图片 + 同名 rsml）。"""
    groups, junk = {}, []
    for path in sorted(p for p in src.iterdir() if p.is_file()):
        if path.name.lower() in JUNK_FILES:
            junk.append(path)
            continue
        groups.setdefault(path.stem, []).append(path)
    return groups, junk


def group_kind(files):
    """判断一组的配对情况：'pair' 图片+rsml 齐全 / 'no_rsml' 缺标注 / 'no_image' 缺图片。"""
    has_img = any(f.suffix.lower() in config.IMAGE_EXTS for f in files)
    has_rsml = any(f.suffix.lower() == ".rsml" for f in files)
    if has_img and has_rsml:
        return "pair"
    return "no_rsml" if has_img else ("no_image" if has_rsml else "other")


def split_counts(n, ratios):
    """把 n 组按比例切成三份：四舍五入后余数留给 train，保证三份之和恰为 n。"""
    n_val = min(int(n * ratios["val"] + 0.5), n)
    n_test = min(int(n * ratios["test"] + 0.5), n - n_val)
    return {"train": n - n_val - n_test, "test": n_test, "val": n_val}


def write_record(out_dir, src, ratios, seed, assign):
    """在输出目录写 split.txt，记录比例/种子/每组归属，便于复现和核对。"""
    lines = [
        "# 数据集划分记录",
        f"源文件夹：{src}",
        f"输出目录：{out_dir}",
        f"比例：train {ratios['train']:g} / test {ratios['test']:g} / val {ratios['val']:g}"
        f"    随机种子：{seed}    时间：{naming.timestamp()}",
        "",
    ]
    for name in SUBSETS:
        names = assign[name]
        lines.append(f"## {name}（{len(names)} 组）")
        lines.append("、".join(names) if names else "（空）")
        lines.append("")
    (out_dir / "split.txt").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    src = Path(args.dir).expanduser()
    if not src.is_dir():
        print(f"[错误] 文件夹不存在：{src}")
        return 1

    ratios, err = resolve_ratios(args)
    if err:
        print(f"[错误] {err}")
        return 1

    groups, junk = collect_groups(src)
    if not groups:
        print(f"[错误] 文件夹里没有文件：{src}")
        return 1
    if junk:
        print(f"[提示] 忽略 {len(junk)} 个系统文件：{'、'.join(f.name for f in junk)}")

    # 侧栏提示：子文件夹不参与划分（与 common/dataset.py 一致，只认当前层）
    subdirs = [d.name for d in src.iterdir() if d.is_dir()]
    if subdirs:
        print(f"[提示] 文件夹内有 {len(subdirs)} 个子文件夹，本工具只划分当前层的文件，子文件夹不动")

    kinds = {name: 0 for name in ("pair", "no_rsml", "no_image", "other")}
    for files in groups.values():
        kinds[group_kind(files)] += 1
    n_files = sum(len(f) for f in groups.values())

    counts = split_counts(len(groups), ratios)
    names = sorted(groups)
    random.Random(args.seed).shuffle(names)  # 先排序再打乱：结果只取决于数据与种子
    assign = {
        "train": names[:counts["train"]],
        "test": names[counts["train"]:counts["train"] + counts["test"]],
        "val": names[counts["train"] + counts["test"]:],
    }

    # ---------- 划分信息 ----------
    print(f"源文件夹：{src}")
    print(f"共 {len(groups)} 组 / {n_files} 个文件"
          f"（图片+rsml 齐全 {kinds['pair']} 组，缺标注 {kinds['no_rsml']} 组，"
          f"缺图片 {kinds['no_image']} 组，其他 {kinds['other']} 组）")
    if kinds["pair"] != len(groups):
        print("[提示] 有组的图片/标注不成对，检查源文件夹是否漏拷了 .rsml 或图片"
              "（训练要求 png 与同名 rsml 同目录）")
    print(f"比例：train {ratios['train']:g} / test {ratios['test']:g} / val {ratios['val']:g}"
          f"    随机种子：{args.seed}")
    for name in SUBSETS:
        ratio = ratios[name]
        if ratio > 0 and counts[name] == 0:
            print(f"[提示] {name} 比例 {ratio:g} 但数据太少（共 {len(groups)} 组），实际分到 0 组")
        if ratio == 0:
            print(f"  {name}：不分配（比例 0）")
        else:
            print(f"  {name}：{counts[name]} 组"
                  f"（{sum(len(groups[n]) for n in assign[name])} 个文件）"
                  f"  {'、'.join(assign[name]) if assign[name] else '空'}")

    if args.dry_run:
        print("\n（预览模式，没有写任何文件）")
        return 0

    # ---------- 落盘 ----------
    out_dir = Path(args.out).expanduser() if args.out else src.parent / f"{src.name}_split"
    if args.out and out_dir.exists():
        print(f"[错误] 输出目录已存在：{out_dir}（换个 --out，或先删掉它，避免与旧结果混在一起）")
        return 1
    out_dir = naming.unique_path(out_dir)

    for name in SUBSETS:
        if not assign[name]:
            continue  # 不分配的子集不建空文件夹
        dst = out_dir / name
        dst.mkdir(parents=True, exist_ok=True)
        for gname in assign[name]:
            for path in groups[gname]:
                target = dst / path.name
                if args.move:
                    shutil.move(str(path), str(target))
                else:
                    shutil.copy2(str(path), str(target))
    write_record(out_dir, src, ratios, args.seed, assign)

    action = "移动" if args.move else "复制"
    print(f"\n完成：{action} {sum(len(groups[n]) for name in SUBSETS for n in assign[name])} 个文件到 {out_dir}")
    print(f"划分记录：{out_dir / 'split.txt'}")
    if not args.move:
        print(f"提示：源文件夹未改动；要把数据接入训练，可把 {out_dir}\\train、{out_dir}\\test "
              f"拷到 datasets\\ 下的 train / test（见 config.py 的 TRAIN_DATA_DIR / TEST_DATA_DIR）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
