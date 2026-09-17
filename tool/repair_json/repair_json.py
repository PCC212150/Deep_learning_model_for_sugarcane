"""检查并修复 labelme json 里 `"imagePath"` 与实际图片文件名不一致的问题。

背景：labelme 的 json 里记着它标注的那张图叫什么名字 ——

    "imagePath": "plant_ S221-4_20251116ST.png"

**这个值应当与该 json 旁边那张图片的文件名（名字 + 后缀）完全相同。**
图片后来被转格式或改名（png → jpg）时，json 里的 imagePath 不会跟着变，
于是 labelme/RootNav 这类软件按 imagePath 回找图片就打不开了。

    改前  "imagePath": "plant_ S221-4_20251116ST.png"
    改后  "imagePath": "plant_ S221-4_20251116ST.jpg"     ← 以磁盘上的实际文件名为准

注意**大小写也算不一致**：Windows 上 `x.JPG` 与 `x.jpg` 都能打开，但训练跑在
Linux 服务器上，那边是区分大小写的 —— 所以按磁盘上的真实写法对齐。

用法：
    python repair_json.py --dir "D:\\目标文件夹" --dry-run     # 先预览，看要改哪些
    python repair_json.py --dir "D:\\目标文件夹"               # 正式修复（就地改）
    python repair_json.py --dir "D:\\目标文件夹" -r            # 递归子文件夹
    python repair_json.py --dir "D:\\目标文件夹" --images "D:\\图片文件夹"
                                                              # json 与图片不在同一目录时

**改动前请先 --dry-run 预览一遍。** 修复是就地改文件（和 add_suffix 一样是有损操作）。

安全性说明（为什么不 json.load → json.dump 写回）：
    `load → dump` 会把**整个文件重写一遍** —— 缩进、空格、浮点写法、转义都可能变，
    `imageData` 里那几 MB 的 base64 也会重新编码一遍。标注数据不该冒这个险。
    本工具只在文本层面替换 `"imagePath"` 后面那个**字符串值本身**，其余字节原样保留，
    改动最小（同 [tool/repair_rsml](../repair_rsml/repair_rsml.py) 的思路）。

改完之后在目标文件夹里生成 repair_json_log.txt，逐条记录
「相对路径 → 旧值 → 新值」，需要时可按它手工改回去。
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

# 只匹配 "imagePath": "…" 后面那个字符串值。
# 前面加否定环视，避免误伤 originalImagePath 这类同样以 imagePath 结尾的键。
# 值内部用 (?:[^"\\]|\\.)* 匹配转义序列 —— base64 里不含引号，不会串到 imageData 上。
IMGPATH_PAT = re.compile(r'((?<![A-Za-z0-9_])"imagePath"\s*:\s*)"((?:[^"\\]|\\.)*)"')

LOG_NAME = "repair_json_log.txt"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="检查/修复 labelme json 里 imagePath 与实际图片文件名不一致的问题",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=('示例：\n'
                '  python repair_json.py --dir "D:\\图片文件夹" --dry-run\n'
                '  python repair_json.py --dir "D:\\图片文件夹"\n'
                '  python repair_json.py --dir "D:\\数据集\\train\\labels\\other" '
                '--images "D:\\数据集\\train\\images"\n'),
    )
    p.add_argument("--dir", required=True, help="目标文件夹（找里面的 .json）")
    p.add_argument("-r", "--recursive", action="store_true", help="递归处理子文件夹")
    p.add_argument("--images", default=None,
                   help="图片所在文件夹；默认与 json 同目录（json 与图片分目录存放时才需要）")
    p.add_argument("--dry-run", action="store_true",
                   help="只打印将要修改的文件和改动内容，不实际写文件")
    p.add_argument("--force", action="store_true",
                   help="连「名字也不同」的也改（默认只报告，因为那多半是 json 配错了图）")
    return p.parse_args(argv)


def read_text(path: Path) -> str:
    """按 utf-8 读，**不做换行转换**（newline=""）—— 否则 \\n 会在写回时变成 \\r\\n，
    等于把整个文件的换行都改了。"""
    with open(path, "r", encoding="utf-8", newline="") as f:
        return f.read()


def write_text(path: Path, text: str):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def get_base(value: str) -> str:
    """取路径里的文件名部分（labelme 可能写 "x.jpg"，也可能写 "images/x.jpg"）。"""
    i = max(value.rfind("/"), value.rfind("\\"))
    return value[i + 1:]


def find_image(stem: str, image_dir: Path):
    """在图片目录里找与 json 同名的图片，返回 (状态, 图片文件名)。

    条件与训练管线一致（common/dataset.py 的 discover_pairs）：**主干精确相等**、
    扩展名属于 config.IMAGE_EXTS。
    """
    cands = sorted(p for p in image_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in config.IMAGE_EXTS
                   and p.stem == stem)
    if not cands:
        return "找不到同名图片（跳过）", None
    if len(cands) > 1:
        # 同一主干两张图（x.png 与 x.jpg 并存）无法判断该指向谁，不猜
        return f"同名图片有 {len(cands)} 张（跳过）", None
    return "ok", cands[0].name


def repair_one(json_path: Path, image_dir: Path, dry_run: bool, force: bool):
    """返回 (状态, 旧值, 新值)。状态见 main() 里的统计口径。"""
    status_img, actual = find_image(json_path.stem, image_dir)
    if actual is None:
        return (status_img, "", "")

    try:
        text = read_text(json_path)
        data = json.loads(text)
    except UnicodeDecodeError as e:
        return ("编码不是 utf-8（跳过）", f"{e}", "")
    except json.JSONDecodeError as e:
        return ("json 解析失败（跳过）", f"{e}", "")
    except OSError as e:
        return ("读文件失败（跳过）", f"{e}", "")

    hits = IMGPATH_PAT.findall(text)
    if not hits:
        return ("缺 imagePath（未改）", "", actual)

    old = data.get("imagePath")
    if not isinstance(old, str):
        return ("imagePath 不是字符串（跳过）", repr(old), actual)

    if old == actual:
        return ("正常", old, actual)          # 逐字节相同（含大小写）才算正常

    # 保留值里的目录前缀，只换文件名部分 —— 改动最小
    base = get_base(old)
    new = old[:len(old) - len(base)] + actual

    if base.lower() == actual.lower():
        kind = "大小写不一致"
    elif Path(base).suffix == "":
        kind = "缺后缀"          # 必须先判：Path("x").stem 还是 "x"，否则会被下一支吃掉
    elif Path(base).stem == Path(actual).stem:
        kind = "后缀不一致"
    else:
        kind = "名字也不同"
        if not force:
            # 名字整体对不上，多半是 json 配错了图（标注内容属于另一株）——
            # 悄悄改名会把这个错误盖掉，所以默认只报告。
            return ("名字也不同（未改，加 --force 才改）", old, new)

    if not dry_run:
        new_text = IMGPATH_PAT.sub(
            lambda m: m.group(1) + json.dumps(new, ensure_ascii=False), text)
        if new_text == text:
            return ("正则没匹配到（跳过）", old, new)
        write_text(json_path, new_text)
    return (f"已修复（{kind}）", old, new)


def main():
    args = parse_args()
    root = Path(args.dir)
    if not root.is_dir():
        sys.exit(f"[错误] 目标文件夹不存在: {root}")
    image_dir = Path(args.images) if args.images else None
    if image_dir is not None and not image_dir.is_dir():
        sys.exit(f"[错误] 图片文件夹不存在: {image_dir}")

    files = sorted(root.rglob("*.json") if args.recursive else root.glob("*.json"))
    if not files:
        sys.exit(f"[错误] {root} 里没有 .json 文件"
                 + ("" if args.recursive else "（子文件夹里的没算，要递归请加 -r）"))

    where = f"图片在 {image_dir}" if image_dir else "图片与 json 同目录"
    print(f"目标: {root}（{'递归' if args.recursive else '只看本层'}；{where}）")
    print(f"找到 {len(files)} 个 .json" + ("   [dry-run：不改任何文件]\n" if args.dry_run else "\n"))

    stat, rows = {}, []
    for f in files:
        idir = image_dir if image_dir is not None else f.parent
        status, old, new = repair_one(f, idir, args.dry_run, args.force)
        stat[status] = stat.get(status, 0) + 1
        rows.append((status, f, old, new))
        if status != "正常":
            mark = "·" if args.dry_run else "✓"
            print(f"  {mark} [{status}] {f.name}")
            if old != new:
                print(f"        {old!r} → {new!r}")

    print("\n统计: " + " | ".join(f"{k} {v}" for k, v in sorted(stat.items())))

    if stat.get("找不到同名图片（跳过）", 0) == len(files) and image_dir is None:
        print("[提示] 一个同名图片都没找到 —— 如果 json 与图片分目录存放"
              "（如 labels/other 与 images），用 --images 指定图片目录")

    changed = [r for r in rows if r[0].startswith("已修复")]
    if args.dry_run and changed:
        print(f"\n[dry-run] 上面这 {len(changed)} 个文件会被修改。去掉 --dry-run 即正式执行。")
    if changed and not args.dry_run:
        log = root / LOG_NAME
        with open(log, "w", encoding="utf-8-sig", newline="") as fh:
            fh.write("# labelme json imagePath 修复记录（旧值 → 新值，需要回滚可按此手工改回）\n")
            wr = csv.writer(fh)
            wr.writerow(["相对路径", "旧 imagePath", "新 imagePath"])
            for _, f, old, new in changed:
                wr.writerow([str(f.relative_to(root)), old, new])
        print(f"已修改 {len(changed)} 个文件，改动记录: {log}")


if __name__ == "__main__":
    main()
