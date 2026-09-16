"""检查并修复 RSML 标注里 `<file-key>` 与文件名不一致的问题。

背景：`<file-key>` 应当等于**该 rsml 文件的主文件名**（不含扩展名），例如

    plant_ S062-2_20251116ST.rsml
      └── 期望 <file-key>plant_ S062-2_20251116ST</file-key>

本项目的约定就是这么写的（见 [inference.py](../../inference.py) 调用
[common/rsml_export.py](../../common/rsml_export.py) 时传的 `file_key=p.stem`）。

**为什么会不一致**：标注是在改名之前做的。用
[tool/add_suffix](../add_suffix/add_suffix.py) 给图片/标注批量加后缀时，
它只改**文件名**，不会动 `.rsml` **内容**里的 `<file-key>` —— 于是文件名变成了
`plant_ S062-2_20251116ST.rsml`，里面却还写着 `plant_ S062-2`。本工具就是它的补丁。

用法：
    python repair_rsml.py --dir "D:\\目标文件夹" --dry-run     # 先预览，看要改哪些
    python repair_rsml.py --dir "D:\\目标文件夹"               # 正式修复（就地改）
    python repair_rsml.py --dir "D:\\目标文件夹" -r            # 递归子文件夹
    python repair_rsml.py --dir "D:\\目标文件夹" --add-missing # 顺带补上完全没有 file-key 的文件

**改动前请先 --dry-run 预览一遍。** 修复是就地改文件（和 add_suffix 一样是有损操作）。

安全性说明（为什么不直接解析 XML 再写回）：
    XML parse → serialize 会把**整个文件重写一遍** —— 属性顺序、自闭合标签写法、
    缩进、编码声明都可能变，标注数据不该冒这个险。本工具只在文本层面替换
    `<file-key>…</file-key>` 中间那一段，**其余字节原样保留**，改动最小。

改完之后在目标文件夹里生成 repair_rsml_log.txt，逐条记录
「文件名 → 旧值 → 新值」，需要时可按它手工改回去。
"""
import argparse
import csv
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape, unescape

# 只匹配 <file-key>…</file-key> 中间那一段，前后都保留原样
KEY_PAT = re.compile(r"(<file-key>)(.*?)(</file-key>)", re.DOTALL)

LOG_NAME = "repair_rsml_log.txt"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="检查/修复 RSML 里 <file-key> 与文件名不一致的问题",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=('示例：\n'
                '  python repair_rsml.py --dir "D:\\图片文件夹" --dry-run\n'
                '  python repair_rsml.py --dir "D:\\图片文件夹"\n'),
    )
    p.add_argument("--dir", required=True, help="目标文件夹路径")
    p.add_argument("-r", "--recursive", action="store_true", help="递归处理子文件夹")
    p.add_argument("--dry-run", action="store_true",
                   help="只打印将要修改的文件和改动内容，不实际写文件")
    p.add_argument("--add-missing", action="store_true",
                   help="对完全没有 <file-key> 的文件，在 <metadata> 末尾补一个"
                        "（默认只报告、不插入）")
    return p.parse_args(argv)


def read_text(path: Path) -> str:
    """按 utf-8 读，**不做换行转换**（newline=""）—— 否则 \n 会在写回时变成 \r\n，
    等于把整个文件的换行都改了。"""
    with open(path, "r", encoding="utf-8", newline="") as f:
        return f.read()


def write_text(path: Path, text: str):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def repair_one(path: Path, dry_run: bool, add_missing: bool):
    """返回 (状态, 旧值, 新值)。状态见 main() 里的统计口径。"""
    expected = path.stem                       # 期望值 = 文件名去扩展名，**空格保留**
    try:
        text = read_text(path)
    except UnicodeDecodeError as e:
        return ("编码不是 utf-8（跳过）", f"{e}", "")
    except OSError as e:
        return ("读文件失败（跳过）", f"{e}", "")

    hits = KEY_PAT.findall(text)
    if not hits:
        if not add_missing:
            return ("缺 <file-key>（未改）", "", expected)
        # 补一个：插在 </metadata> 之前，保持与其它字段同样的缩进
        m = re.search(r"([ \t]*)</metadata>", text)
        if not m:
            return ("缺 <file-key> 且找不到 </metadata>（跳过）", "", expected)
        indent = m.group(1)
        new = (text[:m.start()] + f"{indent}<file-key>{escape(expected)}</file-key>\n"
               + text[m.start():])
        if not dry_run:
            write_text(path, new)
        return ("补上 <file-key>", "(无)", expected)

    olds = [unescape(h[1]) for h in hits]
    if all(o == expected for o in olds):
        return ("正常", olds[0], expected)

    if not dry_run:
        new = KEY_PAT.sub(lambda m: m.group(1) + escape(expected) + m.group(3), text)
        write_text(path, new)
    return ("已修复", " | ".join(sorted(set(olds))), expected)


def main():
    args = parse_args()
    root = Path(args.dir)
    if not root.is_dir():
        sys.exit(f"[错误] 目标文件夹不存在: {root}")

    files = sorted(root.rglob("*.rsml") if args.recursive else root.glob("*.rsml"))
    if not files:
        sys.exit(f"[错误] {root} 里没有 .rsml 文件"
                 + ("" if args.recursive else "（子文件夹里的没算，要递归请加 -r）"))

    print(f"目标: {root}（{'递归' if args.recursive else '只看本层'}）")
    print(f"找到 {len(files)} 个 .rsml" + ("   [dry-run：不改任何文件]\n" if args.dry_run else "\n"))

    stat, rows = {}, []
    for f in files:
        status, old, new = repair_one(f, args.dry_run, args.add_missing)
        stat[status] = stat.get(status, 0) + 1
        rows.append((status, f, old, new))
        if status not in ("正常",):
            mark = "·" if args.dry_run else "✓"
            print(f"  {mark} [{status}] {f.name}")
            if old != new:
                print(f"        {old!r} → {new!r}")

    print("\n统计: " + " | ".join(f"{k} {v}" for k, v in sorted(stat.items())))

    changed = [r for r in rows if r[0] in ("已修复", "补上 <file-key>")]
    if args.dry_run and changed:
        print(f"\n[dry-run] 上面这 {len(changed)} 个文件会被修改。去掉 --dry-run 即正式执行。")
    if changed and not args.dry_run:
        log = root / LOG_NAME
        with open(log, "w", encoding="utf-8-sig", newline="") as fh:
            fh.write("# RSML <file-key> 修复记录（旧值 → 新值，需要回滚可按此手工改回）\n")
            wr = csv.writer(fh)
            wr.writerow(["相对路径", "旧 file-key", "新 file-key"])
            for _, f, old, new in changed:
                wr.writerow([str(f.relative_to(root)), old, new])
        print(f"已修改 {len(changed)} 个文件，改动记录: {log}")


if __name__ == "__main__":
    main()
