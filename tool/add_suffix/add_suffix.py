"""批量为文件添加后缀：后缀插在扩展名之前，如 plant_ S062-1.png -> plant_ S062-1_20251116ST.png。

用法：
    python add_suffix.py --dir "C:\\Users\\21215\\Desktop\\20251126ST" --add "_20251116ST"

默认只处理 DEFAULT_EXTS 里的扩展名，可用 --exts 覆盖；--recursive 递归子文件夹；
改名是有损操作，建议先加 --dry-run 预览一遍再正式执行。
"""
import argparse
import sys
from pathlib import Path

# 默认处理的扩展名（不区分大小写）：图片、RSML 标注、labelme 标注(json)、txt
# 注意：数据集里图片、labels/roots/*.rsml、labels/other/*.json 三处要**同时**改，
# 否则配对就断了（漏掉 .json 会让茎/检查范围的标注对不上）。
DEFAULT_EXTS = {".png", ".jpg", ".jpeg", ".rsml", ".json", ".txt"}


def parse_exts(text):
    """把 "png,jpg" 或 ".png,.jpg" 解析成 {'.png', '.jpg'}。"""
    return {
        e if e.startswith(".") else "." + e
        for e in (x.strip().lower() for x in text.split(","))
        if e
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="给目标文件夹内指定扩展名的文件批量添加后缀（插在扩展名之前）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            '  python add_suffix.py --dir "C:\\Users\\21215\\Desktop\\20251126ST" --add "_20251116ST"\n'
            "  plant_ S062-1.png  ->  plant_ S062-1_20251116ST.png\n"
            "\n"
            "先预览：加 --dry-run；子文件夹：加 -r；扩展名默认 "
            + ",".join(sorted(DEFAULT_EXTS))
        ),
    )
    parser.add_argument("--dir", required=True, help="目标文件夹路径")
    parser.add_argument("--add", required=True, help="要添加的后缀内容，如 _20251116ST")
    parser.add_argument("--exts", default=None,
                        help="逗号分隔的扩展名，默认 " + ",".join(sorted(DEFAULT_EXTS)))
    parser.add_argument("-r", "--recursive", action="store_true", help="递归处理子文件夹")
    parser.add_argument("--force", action="store_true", help="文件名已带该后缀时也再加一次")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要改名的文件，不实际改名")
    return parser.parse_args(argv)


def collect_files(root, exts, recursive):
    """列出待处理的文件；先整体收集再改名，避免边遍历边修改目录。"""
    entries = root.rglob("*") if recursive else root.iterdir()
    return sorted(p for p in entries if p.is_file() and p.suffix.lower() in exts)


def rename_one(path, suffix, force):
    """给单个文件加后缀，返回 (状态, 相关路径)。

    状态：ok 已改名 / already 文件名已含该后缀（跳过）/ conflict 目标文件已存在（不动）
    """
    if not force and path.stem.endswith(suffix):
        return "already", path
    target = path.with_name(path.stem + suffix + path.suffix)
    if target.exists():
        return "conflict", target
    path.rename(target)
    return "ok", target


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.dir).expanduser()
    if not root.is_dir():
        print(f"[错误] 文件夹不存在：{root}")
        return 1
    if not args.add:
        print("[错误] 后缀内容不能为空")
        return 1

    exts = parse_exts(args.exts) if args.exts else DEFAULT_EXTS
    files = collect_files(root, exts, args.recursive)
    print(f"文件夹：{root}")
    print(f"后缀：{args.add}    扩展名：{','.join(sorted(exts))}    "
          f"{'递归' if args.recursive else '仅当前层'}    共 {len(files)} 个文件")
    if args.dry_run:
        print("（预览模式，不会真的改名）")

    counts = {"ok": 0, "already": 0, "conflict": 0}
    for path in files:
        if args.dry_run:
            # 预览时只判断会不会改，不落盘
            status = "already" if (not args.force and path.stem.endswith(args.add)) else "ok"
            if status == "ok" and path.with_name(path.stem + args.add + path.suffix).exists():
                status = "conflict"
        else:
            status, _ = rename_one(path, args.add, args.force)
        counts[status] += 1

        if status == "ok":
            print(f"[改名] {path.name}  ->  {path.stem}{args.add}{path.suffix}")
        elif status == "already":
            print(f"[跳过] {path.name}  （文件名已含该后缀）")
        else:
            print(f"[跳过] {path.name}  （目标已存在：{path.with_name(path.stem + args.add + path.suffix).name}）")

    print(f"\n完成：改名 {counts['ok']} 个，跳过 {counts['already'] + counts['conflict']} 个"
          f"（已含后缀 {counts['already']}，目标重名 {counts['conflict']}）")
    if counts["conflict"]:
        print("提示：目标重名说明文件夹里混有已改过名的文件，请先核对再处理。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
