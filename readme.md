# 甘蔗根系 U-Net 分割与根系统计

目的：构建一个 U-Net 网络，识别并分割甘蔗根系，统计根数量、各根系长度（按像素）和总根系长度

## 数据说明

1. 数据集位于当前项目下的 [datasets](datasets)，已按训练/测试分好两个子文件夹（图片与标注同名同目录，
   如 `plant_ S062-1_20251126ST.jpg` 与 `plant_ S062-1_20251126ST.rsml`，不区分 images/labels 子目录）：

   - [datasets/train](datasets/train)：24 组
   - [datasets/test](datasets/test)：6 组

   两批标注**口径不同**，看结果时必须分开解读：

   | 批次 | 文件名后缀 | 组数（train / test） | 标注内容 |
   | --- | --- | --- | --- |
   | 旧批次 | `_20251116ST` | 18 / 4 | 只标主根，**没有侧根** |
   | 新批次 | `_20251126ST` | 6 / 2 | 主根 + 侧根（嵌套结构，见下） |

2. RSML 标注格式（新批次）：根可以**嵌套**，子根（侧根）写在父根元素里面。

```xml
<plant ID="2" label="barley">
  <annotations>annotation</annotations>
  <root ID="2.1" label="primary">          <!-- 主根 -->
    <geometry>
      <rootnavspline controlpointseparation="50" tension="0.5">
        <point x="1290" y="2209" />
        <point x="1290" y="2263" />         <!-- 分叉点：子根就挂在这个控制点上 -->
        <point x="1298" y="2300" />
      </rootnavspline>
    </geometry>
    <root ID="2.1.1" label="secondary">    <!-- 二级根（侧根），嵌套在父根里 -->
      <geometry>
        <rootnavspline controlpointseparation="50" tension="0.5">
          <point x="1290" y="2263" />       <!-- 起点 = 父根折线上的分叉点 -->
          <point x="1423" y="2238" />
        </rootnavspline>
      </geometry>
    </root>
  </root>
</plant>
```

   - 实测嵌套最深两层（`primary` + `secondary`），子根还能再嵌套子根（ID 形如 `2.1.1.1`）；
   - **子根的第一个 `<point>` 精确等于父根折线上的某个控制点**（30 个标注文件里 293 条子根全部如此）；
   - 每张图可有多个 `<plant>`；含几何的根（主根与侧根**都**计入）全部参与统计；
   - 无坐标（没有 `<geometry>`）的 plant/root 直接忽略：

```xml
<plant ID="1" label="barley">
  <annotations>annotation</annotations>
</plant>
```

## 运行环境

- Python：Anaconda 虚拟环境 **pcc**（Python 3.12）
  激活方式：`conda activate pcc`（或直接调用 `C:\Users\21215\.conda\envs\pcc\python.exe`）
- 依赖已装好：torch 2.12.1+cu130 / torchvision / numpy / pillow / scikit-image（见 [requirements.txt](requirements.txt)）
- 本机 GPU：RTX 5060 Laptop 8G（默认按此配置：长边 1024、batch 2）
- 若部署到 4xRTX3090 服务器：加参数 `--size 1536 --batch 8` 即可，无需改代码

## 使用说明

统一在项目根目录 `D:\python projects\Deep_learning_model_for_sugarcane` 下运行。

1）**训练模型**：

```
python train\train.py
```

可加参数：`--size`（输入长边像素，默认 1024）、`--batch`（默认 2）、`--epochs`（默认 200，验证 Dice 连续 60 轮不提升自动早停）、`--lr`（默认 1e-3）、`--val-size`（从训练集抽几组做验证，默认 2）。
每轮输出：轮次、损失（BCE+Dice）、验证 Dice、本轮耗时(s)。

> **训练侧根时建议加 `--val-size 3`**：按默认种子 42，`--val-size 2` 抽到的验证组是
> `S068-3_20251116ST`、`S221-2_20251116ST`，两张**都是不含侧根的旧标注** —— 那么早停与「保存最优轮」
> 完全看不见侧根学得好不好。`--val-size 3` 会抽到 `S062-1_20251126ST`（含 13 条侧根），
> 训练组仍有 21 组（其中 5 组含侧根）。

2）**测试模型**：

```
python test\test.py                     # 不带参数 = 自动使用最新模型
python test\test.py --model model_202609091135
python test\test.py --model_202609091135   # 兼容写法
```

在 [datasets/test](datasets/test)（每组含 RSML 真值）上评测，输出：像素准确率、IoU、Dice（掩码级平均），
并附每张图"预测根数/总长 vs RSML 真值"的对比，以及**主根、侧根分开的条数与长度**和各自的平均绝对误差，
最后给出测试总耗时。

3）**使用模型（对任意图片文件夹推理）**：

```
python inference.py --model model_202609091135 --dir "D:\目标图片文件夹的路径"
python inference.py --dir "D:\目标图片文件夹的路径"   # 模型省略 = 自动用最新
python inference.py --dir "D:\..." --flat            # RSML 退回旧扁平格式（见下）
```

文件夹里可有多张图片（png/jpg 等），逐张预测并统计。

4）**扫描统计参数（可选）**：见 [tool/tune_stats](tool/tune_stats/readme.md)，
在带真值的数据集上把 `低阈值 × 剪枝长度 × 最短根长 × 计数归一` 扫一遍，按误差挑最优组合。

## 结果输出

1）训练的模型保存至 `D:\python projects\Deep_learning_model_for_sugarcane\model\model_YYYYMMDDHHMM\`
（名称按当前 年月日 时分 创建，如 `model_202609091135`）。该文件夹内除模型参数外还保存训练参数日志：

- `model_202609091135.pth` —— 验证集最优轮权重
- `model_202609091135_log.txt` —— 训练日志（超参与每轮 轮次/损失/验证Dice/耗时）
- `hparams.json` —— 本次训练的全部参数

2）测试模型时调用上述 `.pth`，测试结果保存至对应模型文件夹内，如：
`model\model_202609091135\model_test_{当前 年月日时分}.txt`
内容：每张图的 IoU / Dice / 像素准确率，预测根数/总长/主根数/侧根数与 RSML 真值的对比，
以及汇总的平均绝对误差（根总数、主根数、侧根数、总长、主根长、侧根长）。

3）使用模型时，结果保存至 `D:\python projects\Deep_learning_model_for_sugarcane\result\{目标图片文件夹的名称}\`，内含：
- 一个 txt（与文件夹同名），每行**前四列与旧版完全一致**（空格分隔），其后追加四列：
  **预测的图片名称　根数量　各根系长度(像素，逗号分隔)　总根系长度(像素)　主根数　侧根数　主根总长(像素)　侧根总长(像素)**

  例（列 3 这里省略了中间的长度值，文件里是完整列表）：
  `plant_ S068-1_20251116ST.png 15 1969.4,…,68.6 12260.1 13 2 9687.7 1734.7`
- `{图片名}_mask.png` —— 该图的预测二值掩码（黑底白根）
- `{图片名}_overlay.png` —— 该图的原图 + 预测区域红色半透明叠加（便于目视检查分割效果）
- `{图片名}.rsml` —— 该图的预测根系折线，**与标注 RSML 同格式**（折线点间距约 50px）：
  每条主根一个 plant，长在它上面的侧根**嵌套**成 `secondary`（ID 形如 `1.1` / `1.1.1`），
  子根首点与父根折线上的分叉点重合 —— 结构与新批次标注一致，可直接用 RootNav / rsml-visualizer
  打开查看，或与标注文件对比。加 `--flat` 可退回旧格式（每条折线一个 plant、全部 primary）。

## 主根 / 侧根 的口径与实测精度（重要）

**标注侧**：`label="primary"` 且 ID 只有一层（如 `2.1`）＝ 主根；其余（嵌套的 `secondary`）＝ 侧根。
解析器 [common/rsml_parse.py](common/rsml_parse.py) 已把嵌套展开成平表，统计时按 label/层级分组。

**预测侧**：折线的某个端点落在**另一条折线**的路径上（= 长在分叉点上）就是侧根，那条折线是它的父根；
两端都自由的折线是主根。实现见 [common/root_hierarchy.py](common/root_hierarchy.py)，默认参数：

- `attach_tol = max(6, 0.25×抽稀间距) = 12.5px`（折线按 50px 抽稀，分叉点常落在两个控制点之间，
  弦到真点可达 ~12px，所以容差不能只给几像素）；
- 不许挂在父根自己两端 25px 以内（否则会把「断口」认成侧根）；
- 与父根切向夹角小于 25° 视为「同一条根的续接」而非侧根（实测标注里侧根与父根夹角中位数 68°，
  仅 3.4% 小于 25°，所以基本不会误伤真侧根）；
- 判定不出来的**一律退化为主根**，不丢根、总长守恒（`主根总长 + 侧根总长 ≡ 总数`，代码里有断言级校验）。

**实测精度（2026-09-13，6 张测试图）**：

| 指标 | 结果 |
| --- | --- |
| 主根数平均绝对误差 | 3.8 根（真值平均 16 根）—— 可用 |
| 侧根数平均绝对误差 | 18.7 根（真值平均 17 根）—— **偏得厉害，不能当准数** |

侧根偏低的根因是**侧根太短、模型学不出来**：新批次标注的侧根长度中位数只有 52px、最短 13px
（旧批次几乎没有短于 60px 的根），在 1024 输入下侧根只有约 10px。实测标注侧根「远半段」被预测出
高概率的比例：旧模型 18/102，重训后（24 组，其中 5 组含侧根）27/102。

还有一个**后处理层面的上限**：`config.PRED_SPUR_LENGTH` 的剪枝，以及
[common/skeleton_stats.py](common/skeleton_stats.py) 里的「计数归一」（把过碎的短轨迹拼回现有轨迹），
后者对只有 1 个自由端的侧根特别不利。在**真值掩码**上实测（与模型无关）：

| 计数归一 | 折线数 | 侧根尖端成为折线端点的比例 | 总长 |
| --- | --- | --- | --- |
| 开（默认） | 22 | 19/36 | 8948px（与真值 9008px 相当） |
| 关 | 42 | 35/36 | 8948px（**完全相同**） |

即：关掉归一会保住侧根，但会把互相贴着的根拆碎（无侧根的图上折线数 23 → 42）。两种口径都可用
[tool/tune_stats](tool/tune_stats/readme.md) 扫（`--normalize-count both`），按需要选。

**关于「重调剪枝参数」（2026-09-13 已做）**：在 24 张训练图上扫了 90 组
（low 0.05~0.20 × spur 15~60 × min_len 15~30 × 计数归一 on/off），结论是
**当前 config 这组已在最优附近**（根总数MAE 10.2、主根数MAE 2.8、侧根数MAE 9.7），
把 `PRED_SPUR_LENGTH` 调小到 15~45 反而让主根数误差涨到 5.9~7.1 根 ——
零碎的骨架段会被当成根。所以**没有改默认值**，结果留在
`result/tune_stats-1/tune_stats_*.txt` 与 [config.py](config.py) 的注释里。

**结论：侧根数当前只能当趋势看；要把侧根统计做实，需要更多带侧根的标注（当前仅 8 组）
与更高分辨率的训练。**

## 补充说明

在创建文件或文件夹时，如果名称重复，即在文件末尾添加 `-1`，再重复则 `-2`，以此类推（模型文件夹、测试结果文件、推理结果文件夹/文件均适用）。
