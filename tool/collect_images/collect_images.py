"""把一个文件夹（含各级子文件夹）里的图片全部收集到指定文件夹（没有则自动创建）。

典型场景：网盘下载下来的一批素材，`BaiduNetdiskDownload` 下有一级子文件夹、一级里又套着
二级子文件夹，图片散在最里层。本工具递归扫出所有图片，平铺收拢到一个目标文件夹，
便于后续统一处理（推理、拆数据集、改名等）。

重名处理：不同子文件夹里的同名文件，按项目规范追加 -1、-2 …（规则同 common/naming.unique_path），
磁盘上已有的同名文件不会被覆盖。因为改名后只看文件名已经认不出出处，所以同时在目标文件夹里
生成 collect.txt，逐条记录「目标文件名 <- 原路径」。

用法：
    python collect_images.py --dir "E:\\baiduwangpan\\DownLoad\\BaiduNetdiskDownload" --out "D:\\数据\\全部图片"
    python collect_images.py --dir "..." --out "..." --dry-run          # 只预览，不写任何文件
    python collect_images.py --dir "..." --out "..." --move             # 移动（默认复制，源文件保留）
    python collect_images.py --dir "..." --out "..." --exts .png,.rsml  # 只收指定后缀

注意：移动模式（--move）下源文件夹里的空目录会留下来，工具不负责删空壳。
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from common import naming  # noqa: E402

# 顺手的垃圾文件，不当图片收
JUNK_FILES = {".ds_store", "thumbs.db", "desktop.ini"}

PROGRESS_EVERY = 200   # 每处理这么多文件报一次进度
PREVIEW_LIMIT = 10     # 预览模式最多列多少条


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="递归收集一个文件夹里的所有图片，平铺到一个目标文件夹（重名追加 -1、-2）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            '  python collect_images.py --dir "E:\\baiduwangpan\\DownLoad\\BaiduNetdiskDownload" '
            '--out "D:\\数据\\全部图片"\n'
            '  python collect_images.py --dir "D:\\下载\\素材" --out "D:\\数据\\全部图片" --dry-run\n'
            "\n"
            "目标文件夹不存在会自动创建；已存在则与新收集的文件合并，同名的不覆盖（加 -1、-2）。\n"
            "每次运行都会在目标文件夹写一份 collect.txt，记录目标文件名和原路径的对应关系。"
        ),
    )
    parser.add_argument("--dir", required=True, help="源文件夹（会递归扫到所有子文件夹）")
    parser.add_argument("--out", required=True, help="目标文件夹，不存在则自动创建")
    parser.add_argument("--exts", default=None,
                        help="要收集的后缀，逗号分隔，默认按 config.IMAGE_EXTS："
                             + ",".join(sorted(config.IMAGE_EXTS)))
    parser.add_argument("--move", action="store_true", help="移动文件（默认复制，源文件保留）")
    parser.add_argument("--dry-run", action="store_true", help="只预览收集结果，不写任何文件")
    return parser.parse_args(argv)


def parse_exts(text):
    """'.PNG, jpg' -> {'.png', '.jpg'}；返回 (后缀集合, 错误信息)。"""
    if text is None:
        return set(config.IMAGE_EXTS), None
    exts = set()
    for part in text.replace("，", ",").split(","):
        part = part.strip().lower()
        if part:
            exts.add(part if part.startswith(".") else "." + part)
    if not exts:
        return None, "--exts 是空的，至少要给一个后缀（如 --exts .png,.jpg）"
    return exts, None


def is_inside(path, folder) -> bool:
    """path 是否在 folder 里面（纯字符串比较，不查磁盘；Windows 下忽略大小写）。"""
    p = os.path.normcase(os.path.abspath(path))
    f = os.path.normcase(os.path.abspath(folder))
    return p == f or p.startswith(f + os.sep)


def scan(src, exts, out_dir):
    """递归扫出源文件夹里的目标文件，返回 (文件列表, 跳过的垃圾文件数, 跳过的目标夹内文件数)。

    目标文件夹若正好在源文件夹里面，必须把它排除，否则会把上一次收集进去的文件
    又当成源文件收一遍（越收越多）。
    """
    files, junk, in_out = [], 0, 0
    skip_out = is_inside(out_dir, src)
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        if path.name.lower() in JUNK_FILES:
            junk += 1
            continue
        if skip_out and is_inside(path, out_dir):
            in_out += 1
            continue
        if path.suffix.lower() in exts:
            files.append(path)
    return files, junk, in_out


def pick_name(dst_dir, name, used):
    """给一个源文件名在目标文件夹里挑一个不冲突的名字，返回 (新名字, 是否改过名)。

    规则与 common/naming.unique_path 一致（重名追加 -1、-2 …）。这里没用它是因为还要
    兼顾「本次运行内已经占用」的名字：--dry-run 不落盘，光看磁盘会把同一批里的重名漏判。
    """
    stem, suffix = Path(name).stem, Path(name).suffix
    cand, k = name, 0
    while cand in used or (dst_dir / cand).exists():
        k += 1
        cand = f"{stem}-{k}{suffix}"
    used.add(cand)
    return cand, k > 0


def write_record(out_dir, src, exts, moved, rows, failed):
    """在目标文件夹写 collect.txt：本次收集的每个「目标文件名 <- 原路径」，以及失败清单。"""
    lines = [
        "# 图片收集记录",
        f"源文件夹：{src}",
        f"目标文件夹：{out_dir}",
        f"后缀：{'、'.join(sorted(exts))}",
        f"方式：{'移动' if moved else '复制'}        时间：{naming.timestamp()}",
        f"共 {len(rows)} 个文件（改名 {sum(1 for r in rows if r[2])} 个，失败 {len(failed)} 个）",
        "",
        "## 目标文件名 <- 原路径",
    ]
    for name, path, renamed, _rel in rows:
        mark = "  (重名改名)" if renamed else ""
        lines.append(f"{name} <- {path}{mark}")
    if failed:
        lines.append("")
        lines.append(f"## 失败（{len(failed)} 个）")
        for path, reason in failed:
            lines.append(f"{path}：{reason}")
    path = naming.unique_path(out_dir / "collect.txt")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv=None):
    args = parse_args(argv)
    src = Path(args.dir).expanduser()
    if not src.is_dir():
        print(f"[错误] 源文件夹不存在：{src}")
        return 1

    exts, err = parse_exts(args.exts)
    if err:
        print(f"[错误] {err}")
        return 1

    out_dir = Path(args.out).expanduser()
    if out_dir.exists() and not out_dir.is_dir():
        print(f"[错误] 目标路径已存在且不是文件夹：{out_dir}")
        return 1

    files, junk, in_out = scan(src, exts, out_dir)

    # ---------- 收集信息 ----------
    print(f"源文件夹：{src}")
    print(f"目标文件夹：{out_dir}"
          + ("（已存在，新文件与它合并、同名不覆盖）" if out_dir.exists() else "（不存在，将创建）"))
    print(f"后缀：{'、'.join(sorted(exts))}")
    if junk:
        print(f"[提示] 跳过 {junk} 个系统文件（如 Thumbs.db、.DS_Store）")
    if in_out:
        print(f"[提示] 跳过目标文件夹内的 {in_out} 个文件（目标夹在源文件夹里时不重复收）")
    if not files:
        print(f"[错误] 没有找到要收集的文件（后缀 {'、'.join(sorted(exts))}）")
        return 1

    dirs = sorted({f.parent for f in files})
    shown = "、".join(str(d.relative_to(src)) or "." for d in dirs[:3])
    print(f"找到 {len(files)} 个文件，分布在 {len(dirs)} 个子文件夹里"
          + (f"（如 {shown}{' …' if len(dirs) > 3 else ''}）" if len(dirs) > 1 else ""))

    # ---------- 预览 ----------
    if args.dry_run:
        print(f"\n预览（最多列 {PREVIEW_LIMIT} 条）：")
        used = set()
        for path in files[:PREVIEW_LIMIT]:
            name, renamed = pick_name(out_dir, path.name, used)
            mark = "  (重名改名)" if renamed else ""
            print(f"  {name} <- {path.relative_to(src)}{mark}")
        if len(files) > PREVIEW_LIMIT:
            print(f"  …（还有 {len(files) - PREVIEW_LIMIT} 个未列出）")
        print("\n（预览模式，没有写任何文件）")
        return 0

    # ---------- 落盘 ----------
    out_dir.mkdir(parents=True, exist_ok=True)
    used, rows, failed = set(), [], []
    renamed = done = 0
    for path in files:
        name, was_renamed = pick_name(out_dir, path.name, used)
        target = out_dir / name
        try:
            if args.move:
                shutil.move(str(path), str(target))
            else:
                shutil.copy2(str(path), str(target))
        except OSError as exc:
            failed.append((path.relative_to(src), exc.strerror or str(exc)))
            used.discard(name)
            continue
        rows.append((name, path.relative_to(src), was_renamed, path))
        renamed += was_renamed
        done += 1
        if done % PROGRESS_EVERY == 0 and done < len(files):
            print(f"  已处理 {done}/{len(files)} …", flush=True)

    record = write_record(out_dir, src, exts, args.move, rows, failed)

    action = "移动" if args.move else "复制"
    print(f"\n完成：{action} {done} 个文件到 {out_dir}"
          + (f"（其中 {renamed} 个因重名改成 -1、-2 …）" if renamed else ""))
    print(f"对应关系记录：{record}")
    if failed:
        print(f"[错误] {len(failed)} 个文件处理失败（详见记录文件），例如：")
        for path, reason in failed[:5]:
            print(f"  {path}：{reason}")
        return 1
    if not args.move:
        print(f"提示：源文件夹未改动；要清理源文件可确认无误后用 --move 重跑，或手动删除")
    return 0


if __name__ == "__main__":
    sys.exit(main())
