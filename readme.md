# 甘蔗根系 U-Net 分割与根系统计

目的：构建一个 U-Net 网络，识别并分割甘蔗根系，统计根数量与总根长（按像素）；
同时识别**茎横截面**与**检查范围**，把根系统计限定在检查范围内。

> 2026-09-14 起：模型输出由「根系二分类」改为**三通道多标签**（根系 / 茎横截面 / 检查范围）；
> 推理结果由 txt 改为 **csv**（Excel 可直接打开）；并且**不再区分一级/二级根**。
> 旧的主根/侧根口径与实测结论保留在 [experimental_report/实验报告.md](experimental_report/实验报告.md)。

## 数据说明

数据集位于 [datasets/root](datasets/root)，train / test 结构相同，**图片与标注分目录存放**：

```
datasets/root/train/                  datasets/root/test/
├── images/                           ├── images/
│   ├── plant_ S062-1_20251116ST.png  │   └── …（png / jpg 可混用）
│   └── …                             └── labels/
└── labels/                               ├── roots/   # 根系 RSML（与图片同名）
    ├── roots/                            └── other/   # 茎/检查范围（与图片同名）
    └── other/
```

| 目录 | 内容 | 说明 |
| --- | --- | --- |
| `images/` | 原图（5472×3648，png/jpg） | 配对与统计都以文件名（stem）为准 |
| `labels/roots/*.rsml` | 根系折线标注 | **必须有**，没有配对标注的图片会被跳过 |
| `labels/other/*.json` | labelme 标注 | 可缺；缺了则该图的「茎/检查范围」两通道不参与训练 |

两份标注的对应关系（以 `plant_ S062-1_20251116ST` 为例）：

- `labels/roots/plant_ S062-1_20251116ST.rsml` —— 根系，每条根一段折线（RootNav/RSMLGenerator 格式）；
- `labels/other/plant_ S062-1_20251116ST.json` —— labelme 格式，含两类 shape：
  - `stem`（polygon）：**甘蔗茎的横截面**（图中间那个褐色截面）；
  - `check_background`（rectangle）：**框选检查的范围**（托盘内框区域），根系统计只看框内。

当前规模：**train 24 组 / 18 个植株，test 7 组 / 5 个植株**，两边都三类标注齐全。
测试集按**植株整株划出**，与训练集**植株级零重叠**（2026-09-14 重划）：
`S062-1`（2 个时点）、`S068-1`（2 个时点）、`S001-4`、`S002-2`，
外加 `root_C001-1_20241229CK`（另一批拍摄，用来看跨批次泛化）。

**训练真值的线宽**：RSML 折线按 `config.MASK_LINE_WIDTH`（默认 10px，原图尺度）画成掩码。
这个值要跟图像里根的实际宽度对齐 —— 实测原图根宽中位 10px，而模型分割出来的是「看得见的根」
（宽 12~14px）；真值若只给 5px 细中心线，模型反而会因为"画粗了"被 Dice 扣分
（同一份预测，真值线宽 5px→10px，测试 Dice 0.50→0.62）。线宽对根数/总长统计几乎无影响
（5~13px 实测根数 18.5→17.8、总长 17687→16743px），不用担心贴近的根粘成一根。

- 早先 test 的 6 张与训练集是**逐字节相同的同一批图**（测试指标会偏高），重划时已把训练集里的
  重复件移出；这些副本连同早期换下来的图都在 `datasets/_backup_dup_20260914/`，确认无误后可删。
- 训练/验证划分同样**按植株整组进出**（同一植株的不同时点不会分处两侧），见
  [train/train.py](train/train.py) 的 `split_by_plant`；`config.VAL_SIZE` 是验证**植株数**（当前 4 ≈ 6 张 ≈ 25%）。

### RSML 标注格式

```xml
<plant ID="2" label="barley">
  <annotations>annotation</annotations>
  <root ID="2.1" label="primary">          <!-- 根（本项目不再区分 primary/secondary -->
    <geometry>                              <!-- 解析后一律按「一条根」统计） -->
      <rootnavspline controlpointseparation="50" tension="0.5">
        <point x="1290" y="2209" />
        <point x="1290" y="2263" />
      </rootnavspline>
    </geometry>
    <root ID="2.1.1" label="secondary">    <!-- 子根可能嵌套在父根里 -->
      <geometry>…</geometry>
    </root>
  </root>
</plant>
```

解析时把嵌套展开成平表：**所有含几何的 root（不论层级）都各算一条根**；
无坐标（没有 `<geometry>`）的 plant/root 直接忽略。

## 运行环境

- Python：Anaconda 虚拟环境 **pcc**（Python 3.12）
  激活方式：`conda activate pcc`（或直接调用 `C:\Users\21215\.conda\envs\pcc\python.exe`）
- 依赖已装好：torch 2.12.1+cu130 / torchvision / numpy / pillow / scikit-image（见 [requirements.txt](requirements.txt)）；
  写 CSV 用的是标准库 `csv`，**没有新增依赖**
- 本机 GPU：RTX 5060 Laptop 8G（默认按此配置：长边 1024、batch 2、BatchNorm，见下）
- 若部署到服务器（两张 RTX3090，各 24G）：可提高分辨率与 batch，无需改代码 ——
  `python train/train.py --size 1536 --batch 4 --workers 8`
  （**只会用其中一张卡**：项目里没有 DataParallel/DDP，`export CUDA_VISIBLE_DEVICES=0,1`
  也不能让它用上第二张，见 [config.py](config.py) 的说明）

## 使用说明

统一在项目根目录 `D:\python projects\Deep_learning_model_for_sugarcane` 下运行。

1）**训练模型**：

```
python train\train.py
```

**不传任何参数就是本机推荐配置**（[config.py](config.py)：`MAX_SIDE=1024`、`BATCH_SIZE=2`、`NORM="batch"`）。

可加参数：`--size`（输入长边，默认 1024）、`--batch`（默认 2）、`--accum`（梯度累积，默认 1，
等效 batch = `--batch × --accum`）、`--norm`（`batch`/`group`，默认 batch）、
`--epochs`（轮数上限）、`--lr`、`--patience`（早停轮数，默认 120）、
`--lr-patience`（平台期降 LR 的轮数，默认 25）、
`--val-size`（**验证集植株数**，默认 4；同一植株的所有时点整组进同一侧）。

**学习率调度**：验证 Dice 连续 `LR_PATIENCE` 轮不提升就 `LR × 0.5`（下限 `MIN_LR`），
日志里会打印 `[学习率] …: 1.00e-03 → 5.00e-04`。
**不要改回余弦退火**：`CosineAnnealingLR(T_max=--epochs)` 里的 `T_max` 是「轮数上限」而
不是「实际会跑多少轮」，早停总在上限之前触发，余弦一次都走不到 —— 实测 model_202609151106
第 1 / 83 / 143 轮的 LR 是 0.001000 / 0.000999 / 0.000998，**全程恒定 1e-3**，
val_dice 从第 83 轮起 60 轮纹丝不动（窗口最高 0.4387 < 峰值 0.4413），而训练损失还在降
（0.519→0.497）：典型的「卡在平台上、步长太大下不去」。`LR_PATIENCE` 必须小于 `PATIENCE`
（先降 LR，降完还不行才早停），否则 LR 还没降就被停掉了。

**本机 8G 显存的硬约束**（实测，每训练步耗时 / 峰值显存）：

| 配置 | 每步 | 峰值显存 | 说明 |
| --- | --- | --- | --- |
| `1024 batch2 BatchNorm` | 0.89s | 4.4 GB | **默认，稳** |
| `1536 batch1 BatchNorm` | 0.73s | 4.8 GB | 快，但 batch=1 踩 BatchNorm 的坑（见下） |
| `1536 batch2 BatchNorm` | 13.06s | 9.3 GB | 超 8G → 被 Windows 丢进共享内存，**慢 15 倍** |
| 任意 **GroupNorm** | — | **+4 GB** | 1024/batch2 下 4.4→8.6 GB，本机一律溢出，**不要用** |

**两个必须知道的坑**：

1. **BatchNorm 要求 batch≥2**：batch=1 时每步只用单张图的统计量，与推理用的滑动平均对不上，
   会**严重欠分割**——实测 batch=1 训出的模型评估模式只出 0.36% 前景、训练模式出 1.27%（差 3.5 倍）。
   梯度累积救不了它（只改更新频率，每次前向仍只看到 1 张图）。
2. **GroupNorm 在本机更糟**：它比 BatchNorm 多吃约 4GB，8G 卡一溢出就被丢到共享内存，慢 6~10 倍。
   它适合服务器（24G）上用。

**想提高分辨率请上服务器**（细根在模型里从 1.9px 变 2.9px，是唯一实测支持的方向）：
`--size 1536 --batch 4`（batch≥2 才能用 BatchNorm；显存够也可以 `--norm group`）。

**服务器上务必加 `--workers`**：默认 0 表示读图与数据增强都在主进程里同步做，GPU 会一直等
CPU（表现为「显存占满、`nvidia-smi` 的 GPU-Util 却接近 0」）。`--workers 8` 让加载与训练并行。

**多卡**：本项目是**单卡**代码（`device = torch.device("cuda")` 即 cuda:0），
没有 DataParallel/DDP。不建议为它加多卡 —— `nn.DataParallel` 会让每张卡各自算
BatchNorm 统计量（等效 batch 减半），跟 batch=1 那个坑同源；而且瓶颈是数据量不是算力，
第二张卡留给别的任务更划算。

**两张卡各跑一个训练**（一张 1536、一张 1024）—— `CUDA_VISIBLE_DEVICES` 的作用是
**指定用哪一张**，不是「启用多卡」。设成 1 时物理卡 1 在该进程里就叫 `cuda:0`，
正好是 train.py 会用的那张：

```bash
tmux new -s g0        # 会话 A → 物理卡 0
CUDA_VISIBLE_DEVICES=0 python train/train.py --size 1536 --batch 4 --workers 4
# Ctrl-B 再按 D 脱离；tmux attach -t g0 回去

tmux new -s g1        # 会话 B → 物理卡 1
CUDA_VISIBLE_DEVICES=1 python train/train.py --size 1024 --batch 8 --workers 4
```

- 两个进程都往 `model/` 写，同一分钟启动会得到 `model_YYYYMMDDHHMM` 与 `…-1`
  （`naming.create_unique_dir` 是原子创建，不会撞名崩），靠各自的 `hparams.json`
  里的 `"size"` 区分。后续 `test.py` / `inference.py` **要显式写 `--model <文件夹名>`** ——
  自动「取最新」在两个并发训练下没有意义。
- `--workers` 两边相加别超过 CPU 核数（`nproc`）；16 核就用 `--workers 4`。
- 两个训练默认同一个 `--seed 42` → 训练/验证划分完全相同，可在同一批验证图上直接比。
  但 1536 塞不下 batch 8，两次的有效 batch 不同，所以这是「分辨率+batch」的联合对照，
  不是纯净的分辨率对照。**最终都以 `test.py` 在测试集上的结果为准。**

分辨率必须与推理一致：`test.py` / `inference.py` / `tool/tune_stats` 现在会**自动读取模型训练时的
`--size`**，显式传 `--size` 覆盖时会提示两者不一致。实测 1024 训的模型用 2048 推理，总长误差从 4278px 涨到 9675px。

每轮输出：轮次、损失、验证 Dice（**只看根系通道**，早停与「保存最优轮」都按它）、本轮耗时(s)，
行尾另附茎与检查范围通道的 Dice：

```
[Epoch 012/200] loss=0.4123 val_dice=0.4831 val_iou=0.3176 time=2.6s lr=1.00e-03 root_dice=0.4831 stem_dice=0.7… check_dice=0.9… *best*
[学习率] 验证 Dice 连续 25 轮未提升：1.00e-03 → 5.00e-04
```

**clDice 拓扑损失**（`LOSS_CLDICE_W` / `CLDICE_ITERS`，2026-09-15 新增）：对**骨架**算 Dice，
专门约束连通性。日志里会多一列 `cldice=`（正常在 0.6~0.9 且随训练爬升）：

```
[Epoch 003/5000] … lr=1.00e-03 cldice=0.1383 root_dice=0.1519 stem_dice=… check_dice=… *best*
```

> **`CLDICE_ITERS` 是最容易踩的坑**：软骨架靠「逐层腐蚀、把残差并进骨架」实现，
> **迭代次数必须 ≥ 结构在模型分辨率下的最大半径**，否则骨架退化成 0 —— 而骨架为 0 时
> clDice 恰好等于 1、损失等于 0，**梯度静默消失，等于没开**。实测（等宽条，`iter` 下限）：
>
> | 宽度 | 硬标签 | 软概率图 | 对应 |
> | --- | --- | --- | --- |
> | 2px | 1 | 3 | 1024 下的根（约 1.9px） |
> | 4px | 1 | 5 | 2048 下的根（约 3.7px） |
> | 8px | 3 | 10 | 靠近茎的根部团块 |
>
> 默认 `CLDICE_ITERS = 5` 覆盖 1024/2048 的根。**若日志里 `cldice=` 恒为 1.0000 就是退化了**，
> 那时会打印一条 `[警告] 真值骨架大量为空`，把 iters 调大或把 `LOSS_CLDICE_W` 置 0。
> 代价不小：B=2、1024×688 下前向+反向 it=3 约 0.36s、it=5 约 0.56s（笔记本 RTX 5060），
> 实测每轮从 5.0s 涨到 6.0s；服务器上按批大小线性放大。

**多模型集成**：`--model` 支持逗号分隔，推理时对多个模型的**概率图取平均**再走同一套后处理。

```bash
python test\test.py       --model model_A,model_B,model_C
python inference.py       --model model_A,model_B,model_C --dir "D:\目标文件夹"
python tool\tune_stats\tune_stats.py --dir datasets/root/train --model model_A,model_B
```

多个模型断的地方不一样，平均后碎片更容易连上（实测拿一个旧模型混进去，根数 MAE 12.14→11.00、
总长 MAE 2793→2206，虽然那个旧模型的茎是坏的、把茎指标拖低了）。**集成要求所有模型训练时的
`--size` 一致** —— 概率图必须在同一个网格上平均，不一致会直接报错而不是静默按第一个模型的尺寸跑。
想要真正有效的集成，应该用**同一套配置、不同 `--seed`** 训 3~5 个模型。

**验证 Dice 的噪声很大**（4 植株 / 6 张图，实测平台期相邻轮能摆动 ±0.07），
所以「最优轮」的挑选本身就带随机性 —— 相邻两个 checkpoint 的实际水平差距，
未必真有日志上那 0.01 的差别。

想钉死验证植株，改 [config.py](config.py) 的 `VAL_PLANTS = ("S068-3", "S062-3")`（非空即优先用它）。

2）**测试模型**：

```
python test\test.py                     # 不带参数 = 自动使用最新模型
python test\test.py --model model_202609091135
python test\test.py --model_202609091135   # 兼容写法
```

在 [datasets/root/test](datasets/root/test) 上评测，逐图给出**三类通道各自的 IoU / Dice / 像素准确率**
与「预测根数/总长 vs RSML 真值」的对比，末尾是平均绝对误差与各通道平均指标。
像素指标在模型分辨率上计算（与训练时的验证 Dice 同一口径）。

3）**使用模型（对任意图片文件夹推理）**：

```
python inference.py --model model_202609091135 --dir "D:\目标图片文件夹的路径"
python inference.py --dir "D:\目标图片文件夹的路径"          # 模型省略 = 自动用最新
python inference.py --dir "D:\目标图片文件夹的路径" --mm-per-px 0.1234   # CSV 追加 mm 列
```

文件夹里可有多张图片（png/jpg 等），逐张预测并统计。

4）**扫描统计参数（重训后建议做一次）**：见 [tool/tune_stats](tool/tune_stats/readme.md)。
`PRED_LOW_THRESHOLD` / `PRED_SPUR_LENGTH` / `MIN_ROOT_LENGTH` 这组值是在**旧模型 + 全图统计**
口径下调出来的；改成「三通道模型 + 检查范围限定」后需在训练集上重扫一遍再定稿。

## 结果输出

1）训练的模型保存至 `model\model_YYYYMMDDHHMM\`（名称按当前 年月日 时分 创建）：

- `model_YYYYMMDDHHMM.pth` —— 验证集最优轮权重（含 `out_ch` / `class_names`）
- `model_YYYYMMDDHHMM_log.txt` —— 训练日志（超参与每轮指标）
- `hparams.json` —— 本次训练的全部参数（含训练/验证的植株与图片清单）

2）测试结果保存在对应模型文件夹内：`model\model_YYYYMMDDHHMM\model_test_{年月日时分}.csv`
（UTF-8 BOM，Excel 直接双击打开）。每行一张图，列：

```
图片名 | IoU(根) Dice(根) 像素准确率(根) | IoU(茎) … | IoU(检查) … |
GT根数 预测根数 GT总长(px) 预测总长(px) 预测各根长(px,分号分隔) [GT总长(mm) 预测总长(mm)]
```

文件末尾是若干以 `#` 开头的汇总行（平均指标、MAE、统计口径、耗时），脚本可跳过。

3）推理结果保存在 `result\{目标图片文件夹名}\`，内含：

- `{文件夹名}.csv` —— 每行一张图（UTF-8 BOM）：

  | 图片名 | 根数量 | 起点锚定(条) | 总根长(px) | 平均根长(px) | 最长根(px) | 各根长度(px) | 茎面积(px²) | 检查区面积(px²) | check_ok |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

  例：`plant_ S068-1_20251116ST.png 15 12 12260.1 817.3 1969.4 1969.4;1543.2;… 44231 12345984 是`

  「各根长度」用**分号**分隔（避免与 CSV 的逗号打架）；`check_ok=否` 表示该图的检查范围
  没预测出来、退回全图统计。行尾附 `#` 开头的参数与耗时说明。
- `{图片名}_mask.png` —— 统计口径的根系掩码（已限定在检查范围内，黑底白根）
- `{图片名}_stem.png` —— 识别出的茎横截面掩码
- `{图片名}_check.png` —— 识别出的检查范围掩码
- `{图片名}_overlay.png` —— 原图 + 根系(红) + 茎(橙) + 检查范围(绿框)，一张看完
- `{图片名}.rsml` —— 预测根系折线，**每条折线一个 plant**（不再分主根/侧根），
  起点已锚定到茎（每条根都从茎发出），可直接用 RootNav / rsml-visualizer 打开，或与标注文件对比

## 统计口径（重要）

**根数量与根长只在模型识别出的检查范围内统计**，范围外的一律不计入：

1. 检查范围通道 → 二值化 → 取最大连通域 → 按**行/列覆盖量剖面**拟合成矩形（它本来就是个矩形）。
   这里不能直接取外接矩形：托盘之外的玻璃/水面碎片常与主区域连通，面积只差几个百分点，
   却足以把外接矩形拉到接近整图（实测与标注框的平均 IoU 只有 0.84，改用剖面拟合后 0.98）；
2. 矩形按 `config.CHECK_MARGIN_PX`（默认 10px）**向外膨胀** —— 贴着框边的根被切断的话，
   一根会被统计成两根；
3. 在**概率层**把框外的根系概率清零，再做滞回阈值（顺序反了的话，框外的弱响应会把框内
   两段连通起来）；最后再与矩形精确求交。
4. 若检查范围预测异常（占图面 <5% 或 >99.9%），该图 `check_ok=否`，退回全图统计。

**不扣茎**：茎横截面只作为一类识别结果输出，不影响根系统计（实测只有 0.4% 的标注点落在茎上）。

**起点锚定到茎**：每条预测折线的起点会沿直线补到茎边界，补回的这一段**计入根长**。

- 为什么需要：茎外面套着一圈黑色泡沫/海绵环（固定插穗用），根是从环后面钻出来的；
  那圈在图像上确实不是根，模型判成背景是对的，所以检测到的根离茎有 100~300px 的空档。
- 而标注是**从茎边开始画折线**的（真值里约 31% 的根长集中在距茎 300px 内、只占 2.8% 图面），
  也就是那一段根真实存在、只是被遮住了 —— 补回来才与标注同口径。
- 效果（6 张测试图）：预测总长与真值的平均绝对误差从 **3151px 降到 907px**；
  27 条折线里 24 条的起点落在茎边界 50px 内。
- 阈值 = `STEM_ANCHOR_FACTOR` × 茎等效半径，卡在 `STEM_ANCHOR_MIN_PX`~`STEM_ANCHOR_MAX_PX`
  之间（见 [config.py](config.py)）。两端都超出阈值的折线**原样保留**（当作独立根计入，不丢信息），
  CSV 里的「起点锚定(条)」就是成功锚定的条数。

**茎通道是总根长的命门（2026-09-15 发现并修复）**：锚定依赖茎，茎画不出来 → 锚定 0 条 →
总根长系统性偏短。当时 1024 模型在测试集上 7 张**全部**偏小，其中 2 张完全没检出茎：

| 图片 | GT茎(px) | 预测茎(px) | 起点锚定 | 总长误差 |
| --- | --- | --- | --- | --- |
| S062-1_1116 | 5751 | 2421 | 24 条 | +1% |
| S062-1_1126 | 4609 | 2470 | 22 条 | +6% |
| S068-1_1116 | 4526 | 92 | 6 条 | −6% |
| S068-1_1126 | 4759 | 73 | 4 条 | −24% |
| S001-4 | 7647 | **0** | **0 条** | −21% |
| S002-2 | 6929 | **0** | **0 条** | −33% |
| CK-20241229 | 5217 | 2057 | 40 条 | −14% |

根因：茎只占图面约 0.8%，正负样本比 ~1:100，而 BCE 权重与根系相同、Dice 权重还只有一半
（0.5），是小目标欠学习的典型。修法是把 `LOSS_DICE_W` 的 stem 提到 1.0 并给
`LOSS_POS_WEIGHT` 的 stem 设 10（`config.py`）。**总根长 3592px 的平均绝对误差里一大半是
茎通道的锅，不是根通道的** —— 换模型后如果根数/总长仍然不稳，先看茎画出来没有，别急着调
后处理。

这套口径只在 [common/predict.py](common/predict.py) 里实现一次，`test.py` / `inference.py` /
`tool/tune_stats` 共用 —— 避免「调参时用的口径」和「部署时用的口径」不一致。

**长度单位**：[config.py](config.py) 的 `MM_PER_PX` 默认 0（只输出像素）。要出 mm 列，
需要量一个已知尺寸（如托盘内框实宽）及其在图上的像素宽度，算出「1 像素 = 多少毫米」后填进去，
或在命令行用 `--mm-per-px` 覆盖。注意不同批次/不同物距的图未必同一标定，跨图比较前先确认。

**关于旧模型**：1 通道（只有根系）的旧权重**无法**用于新流程，`inference.py` / `test.py`
会明确报错退出（不会静默降级成只出根系）。请用改造后重新训练得到的权重。

## 补充说明

在创建文件或文件夹时，如果名称重复，即在文件末尾添加 `-1`，再重复则 `-2`，以此类推
（模型文件夹、测试结果文件、推理结果文件夹/文件均适用）。
