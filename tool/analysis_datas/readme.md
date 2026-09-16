# 把推理结果画成折线图（一个编号一张图）

读 inference.py 产出的 CSV（如 `root_pictures.csv`），**按编号分组画折线图**：

- 每个**编号**（如 `C001`）单独一张图；
- 图里每个**重复次序**（`C001-1` / `C001-2` / `C001-3` / `C001-4`）是一条折线；
- 横坐标 = 拍摄日期（20241229、20241231 …，**按真实时间间隔排布**，不是等距序号）；
- 纵坐标 = 总根长 (px)，**从 0 起**（根长是个"量"，截断纵轴会把微小波动放大成陡坡）。

## 输入的文件名格式

工具靠文件名分组，所以格式要保住：

```
root_C001-1_20241229CK.jpg
      └编号┘ └重复┘ └─日期─┘└后缀┘
```

即 `…_{编号}-{重复}_{8位日期}{后缀}.{扩展名}`。解析用 `search` 而不是 `match`，
所以前缀写成 `plant_`、`plant_ `（带空格）、`root_` 都认得。

**图片名里的空格会被去掉**：项目里 `plant_ S062-1_…`（带空格）与 `plant_S001-1_…` 两种
写法并存，本工具按项目约定统一先 `replace(" ", "")` 再解析（同
[common/dataset.py](../../common/dataset.py) 的 `plant_key`）。
清理后的完整 CSV 会另存一份副本到输出目录（`{原名}_无空格.csv`），
**原始 CSV 不会被改动**。

## 数据过滤

`check_ok` / `root_ok` 为「否」的行默认**跳过** —— 这两种情况代表检查范围没识别出来、
或滞回低阈值被背景底噪淹没（见主 readme「背景底噪兜底」），点位不可信，不该进趋势。
想保留就加 `--keep-bad`。

同一「编号 + 重复 + 日期」出现多次时**取最后一次**，并在控制台列出来。

## 用法

在项目根目录下运行：

```
python tool\analysis_datas\analysis_datas.py --csv "C:\Users\me\Desktop\root_pictures.csv"
python tool\analysis_datas\analysis_datas.py --csv a.csv,b.csv          # 多个文件
python tool\analysis_datas\analysis_datas.py                            # 用 analysis_datas\ 下所有 csv
python tool\analysis_datas\analysis_datas.py --csv x.csv --ids C001,C203   # 只画这几个编号
python tool\analysis_datas\analysis_datas.py --csv x.csv --no-overview     # 不生成总览图
```

| 参数 | 说明 |
| --- | --- |
| `--csv` | 输入 CSV，逗号分隔可给多个；省略则用 `analysis_datas\` 下所有 `*.csv` |
| `--out` | 输出根目录，默认 `analysis_datas\`（每个 csv 在其中各建一个同名文件夹） |
| `--ids` | 只画这些编号，逗号分隔 |
| `--dpi` | 单图分辨率，默认 150 |
| `--keep-bad` | 保留 `check_ok` / `root_ok` = 否 的点 |
| `--no-overview` | 不生成 `_总览.png` |

## 输出

`analysis_datas/{csv文件名}/`（重名追加 `-1`、`-2` …，项目规范）：

| 文件 | 内容 |
| --- | --- |
| `{编号}.png` | 每个编号一张折线图 |
| `_总览.png` | 所有编号的缩略图拼成一张，快速扫哪个编号有异常 |
| `_汇总.csv` | 每个编号的日期数、各重复次序的点数、总根长最小/最大值，排查缺数据用 |
| `{原名}_无空格.csv` | 只在原 CSV 里有带空格的图片名时才生成，图片名已去掉空格 |

实测规模：`root_pictures.csv`（3377 行 / 273 个编号）跑完约 **30 秒**。

## 配色与图表规范

配色用 dataviz 规范的前 4 个分类色槽（`#2a78d6` 蓝 / `#eb6834` 橙 / `#1baf7a` 青 /
`#eda100` 黄），跑过规范自带的调色板校验脚本：

```
[PASS] 明度带   [PASS] 彩度下限   [PASS] 色盲相邻对 ΔE 9.1   [PASS] 常视 ΔE 22.9
[WARN] 对比度   青色 2.74 / 黄色 2.11 低于 3:1 → 需要「可见标签兜底」
```

那个 WARN 的兜底就是**图例 + 线端直接标注** —— 身份不靠颜色单独承载。
线端标注做了**纵向去重叠**（相邻标签强制拉开间距），所以不会糊成一团。

**同一编号超过 4 个重复次序时，颜色会从头循环** —— 规范禁止生成新色相，
真超过了应该考虑分面或合并，而不是加颜色。
