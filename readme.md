# 甘蔗根系 U-Net 分割与根系统计

目的：构建一个 U-Net 网络，识别并分割甘蔗根系，统计根数量、各根系长度（按像素）和总根系长度

## 数据说明

1. 数据集位于当前项目下的 [datasets](datasets)，已按训练/测试分好两个子文件夹：
   - [datasets/train](datasets/train)：16 组（每组一张水培甘蔗根系图片 + 同名 RSML 标注）
   - [datasets/test](datasets/test)：6 组
   图片与标注同名同目录（如 `plant_ S062-1.png` 与 `plant_ S062-1.rsml`），不再区分 images/labels 子目录。

2. RSML 标注格式示例（如 [datasets/test/plant_ S062-1.rsml](datasets/test/plant_%20S062-1.rsml)）：

```xml
<plant ID="2" label="barley">
  <annotations>annotation</annotations>
  <root ID="2.1" label="primary">
    <geometry>
      <rootnavspline controlpointseparation="50" tension="0.5">
        <point x="1638" y="786" />
        <point x="1505" y="819" />
        <point x="1447" y="849" />
        <point x="1401" y="857" />
        <point x="1306" y="903" />
        <point x="1260" y="928" />
        <point x="1222" y="1007" />
        <point x="1206" y="1090" />
        <point x="1148" y="1194" />
        <point x="1123" y="1285" />
        <point x="1089" y="1356" />
        <point x="1081" y="1497" />
        <point x="1089" y="1585" />
        <point x="1156" y="1680" />
        <point x="1210" y="1780" />
      </rootnavspline>
    </geometry>
  </root>
</plant>
```

表示标注的一条根（其中 `<point x="1638" y="786" />` 为该根 ID=2.1 的起点）。每张图可有多个 `<plant>`、多条根，所有含几何的根都会计入统计。

3. 对这种没有坐标（无几何）的 plant 直接忽略：

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

可加参数：`--size`（输入长边像素，默认 1024）、`--batch`（默认 2）、`--epochs`（默认 200，验证 Dice 连续 60 轮不提升自动早停）、`--lr`（默认 1e-3）。
每轮输出：轮次、损失（BCE+Dice）、验证 Dice、本轮耗时(s)。

2）**测试模型**：

```
python test\test.py                     # 不带参数 = 自动使用最新模型
python test\test.py --model model_202609091135
python test\test.py --model_202609091135   # 兼容写法
```

在 [datasets/test](datasets/test)（每组含 RSML 真值）上评测，输出：像素准确率、IoU、Dice（掩码级平均），并附每张图"预测根数/总长 vs RSML 真值"的汇总对比，以及测试总耗时。

3）**使用模型（对任意图片文件夹推理）**：

```
python inference.py --model model_202609091135 --dir "D:\目标图片文件夹的路径"
python inference.py --dir "D:\目标图片文件夹的路径"   # 模型省略 = 自动用最新
```

文件夹里可有多张图片（png/jpg 等），逐张预测并统计。

## 结果输出

1）训练的模型保存至 `D:\python projects\Deep_learning_model_for_sugarcane\model\model_YYYYMMDDHHMM\`
（名称按当前 年月日 时分 创建，如 `model_202609091135`）。该文件夹内除模型参数外还保存训练参数日志：

- `model_202609091135.pth` —— 验证集最优轮权重
- `model_202609091135_log.txt` —— 训练日志（超参与每轮 轮次/损失/验证Dice/耗时）
- `hparams.json` —— 本次训练的全部参数

2）测试模型时调用上述 `.pth`，测试结果保存至对应模型文件夹内，如：
`model\model_202609091135\model_test_{当前 年月日时分}.txt`
内容：每张图的 IoU / Dice / 像素准确率，及预测根数、总长与 RSML 真值的对比和平均误差。

3）使用模型时，结果保存至 `D:\python projects\Deep_learning_model_for_sugarcane\result\{目标图片文件夹的名称}\`，内含：
- 一个 txt（与文件夹同名），每行四列（空格分隔），各列分别为：
  **预测的图片名称　根数量　各根系长度(像素，逗号分隔)　总根系长度(像素)**
- `{图片名}_mask.png` —— 该图的预测二值掩码（黑底白根）
- `{图片名}_overlay.png` —— 该图的原图 + 预测区域红色半透明叠加（便于目视检查分割效果）
- `{图片名}.rsml` —— 该图的预测根系折线，**与标注 RSML 同格式**（每条根 = 一个 plant 下的 primary 根，
  折线点间距约 50px），可直接用 RootNav / rsml-visualizer 打开查看，或与标注文件对比

## 补充说明

在创建文件或文件夹时，如果名称重复，即在文件末尾添加 `-1`，再重复则 `-2`，以此类推（模型文件夹、测试结果文件、推理结果文件夹/文件均适用）。
