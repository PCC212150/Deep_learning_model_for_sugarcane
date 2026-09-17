# 修复 RSML 的 `<file-key>`（与文件名对齐）

检查 RSML 标注文件里的 `<file-key>` 是否等于**文件名**（不含扩展名），不一致就改成一致的：

```
plant_ S062-2_20251116ST.rsml
  ✗ <file-key>plant_ S062-2</file-key>              ← 改前
  ✓ <file-key>plant_ S062-2_20251116ST</file-key>   ← 改后
```

## 为什么会不一致

**标注是在改名之前做的。** 用 [tool/add_suffix](../add_suffix/add_suffix.py) 给图片和标注批量
加日期后缀时，它只改**文件名**，不会动 `.rsml` **内容**里的 `<file-key>` —— 于是文件名变成了
`plant_ S062-2_20251116ST.rsml`，里面却还写着 `plant_ S062-2`。

本工具就是 `add_suffix` 的补丁。**2026-09-16 体检你现有数据集的结果**：

| 目录 | 结果 |
| --- | --- |
| `datasets/root/train/labels/roots/` | **24 个全部不一致**（都缺日期后缀） |
| `datasets/root/test/labels/roots/` | 6 个不一致，1 个正常 |

## 影响范围

- **不影响本项目的训练/推理**：`common/` 里只有 *写* `file-key` 的代码
  （[common/rsml_export.py](../../common/rsml_export.py)），没有任何地方 *读* 它。
  配对与统计全靠**文件名**，所以模型指标不受影响。
- **有影响的是外部工具**：RootNav、rsml-visualizer 这类靠 `file-key` 回找对应图片的软件，
  名字对不上就打不开图。

所以：**不急着修，但拿去 RootNav 里看之前要先修。**

## 用法

在项目根目录下运行：

```
python tool\repair_rsml\repair_rsml.py --dir "D:\目标文件夹" --dry-run      # 先预览
python tool\repair_rsml\repair_rsml.py --dir "D:\目标文件夹"                # 正式修复
python tool\repair_rsml\repair_rsml.py --dir "D:\目标文件夹" -r             # 递归子文件夹
python tool\repair_rsml\repair_rsml.py --dir "D:\目标文件夹" --add-missing  # 顺带补上没有 file-key 的
```

| 参数 | 说明 |
| --- | --- |
| `--dir` | 目标文件夹（**必填**） |
| `-r` / `--recursive` | 递归子文件夹（默认只看本层） |
| `--dry-run` | 只打印将要改的内容，不写文件 |
| `--add-missing` | 对完全没有 `<file-key>` 的文件，在 `</metadata>` 前补一个；默认只报告 |

**这是就地修改文件的有损操作，请先 `--dry-run` 预览一遍**（同 [add_suffix](../add_suffix/add_suffix.py)）。

修复后会在目标文件夹里生成 `repair_rsml_log.txt`，逐条记录「相对路径 → 旧值 → 新值」，
需要回滚可以按它手工改回去。

> ⚠️ **日志别留在数据文件夹里**：如果这个文件夹接下来要拿去划数据集
> （[tool/separate_dataset](../separate_dataset/readme.md)），请先把日志挪走 ——
> 划分工具是按「文件夹里所有文件」分组的，一个 `.txt` 会被当成一组文件划进 train/test。

## 为什么是文本替换，不是解析 XML

XML `parse → serialize` 会把**整个文件重写一遍** —— 属性顺序、自闭合标签写法、缩进、
编码声明都可能变。标注数据不该冒这个险。本工具只在文本层面替换
`<file-key>…</file-key>` **中间那一段**，其余字节原样保留。

已经验证过（对样例文件）：

```
原始 5131 字节 -> 改后 5142 字节（file-key 从 13 字变 24 字）
抠掉 <file-key>…</file-key> 后逐字节相同 -> True     ← 证明只动了这一处
CRLF 换行保持原样（读写的 newline="" 不做换行转换）
```

顺带一提：读写都用 `newline=""`，**不做换行转换** —— 否则 Windows 上会把整份文件的
`\n` 悄悄改成 `\r\n`，变成一个巨大的假 diff。

## 边界情况

- **文件名里的空格保留**：`plant_ S062-2_20251116ST` 的 `<file-key>` 就该带那个空格。
  这一点和 [tool/analysis_datas](../analysis_datas/) 相反 —— 那边按项目约定去空格做**解析**，
  这里要跟文件名**逐字对齐**，所以不能去。
- **XML 转义**：文件名里若含 `&`、`<`、`>`，写入时会正确转义成 `&amp;` 等，比较时先反转义。
- **一个文件里出现多个 `<file-key>`**：全部替换成同一个值（正常文件只有一个）。
- **编码不是 utf-8**：跳过并报告，不猜编码。
- **幂等**：已经是正确的文件报「正常」，重复运行不会产生任何改动。
