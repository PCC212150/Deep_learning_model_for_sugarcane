"""项目全局配置：目录路径与默认超参数。

所有可执行脚本（train/train.py、test/test.py、infrernce.py）都会加载本文件；
命令行参数会覆盖这里的默认值。部署到服务器（如 4xRTX3090）时，
优先用命令行 --size/--batch/--epochs 调整，无需改代码。
"""
from pathlib import Path

# 项目根目录（本 config.py 所在目录）
PROJECT_ROOT = Path(__file__).resolve().parent

# ------------------------- 目录 -------------------------
DATASETS_DIR = PROJECT_ROOT / "datasets"
TRAIN_DATA_DIR = DATASETS_DIR / "train"   # 训练：png 与 rsml 同名同目录
TEST_DATA_DIR = DATASETS_DIR / "test"     # 测试：同上
MODEL_DIR = PROJECT_ROOT / "model"        # 模型根目录（内含 model_YYYYMMDDHHMM 子文件夹）
RESULT_DIR = PROJECT_ROOT / "result"      # 推理结果根目录

# ------------------------- 数据 -------------------------
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_LINE_WIDTH = 5       # RSML 折线画掩码线宽(px)，在原图分辨率上绘制（已确认方案）
MAX_SIDE = 1024           # 训练/测试输入长边像素（RTX 5060 8G 配置；服务器可加 --size 1536）
SPUR_LENGTH = 30.0        # 骨架剪枝阈值(px，原图尺度)：短于此长度的末梢视为噪声（当前未使用）

# ---- 预测结果统计（根数/长度）专用后处理参数（按实测调优） ----
# 预测掩码在种子基部常成团块、细根处易断续：
# 滞回阈值把断段接回；剪枝阈值加大以去除团块毛刺（避免一根被数成多根）
PRED_LOW_THRESHOLD = 0.10  # 滞回低阈值（高阈值固定 0.5）
PRED_SPUR_LENGTH = 60.0    # 预测掩码的骨架剪枝阈值(px)
MIN_ROOT_LENGTH = 20.0     # 独立骨架段短于此长度(px)视为噪声忽略
# 2026-09-13 用 tool/tune_stats 在 24 张训练图上重扫（low 0.05~0.20 × spur 15~60 × min_len 15~30
# × 计数归一 on/off 共 90 组，模型 model_202609130020）：**当前这组就是最优附近** ——
# 根总数MAE 10.21、主根数MAE 2.8、侧根数MAE 9.7；把 spur 调小（15~45）反而让主根数误差变大
# （5.9~7.1 根），因为零碎骨架段被当成了根。同时归一 normalize（skeleton_stats 里那步）
# 关掉能救回侧根但会把贴着的根拆碎，需按用途取舍，见 readme「主根/侧根的口径与实测精度」。

# ------------------------- 训练超参 -------------------------
BATCH_SIZE = 2            # RTX 5060 8G 下 batch=2；服务器可 --batch 8
EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 60             # 验证 Dice 连续 N 轮无提升则提前停止
VAL_SIZE = 2              # 从训练数据中固定抽出作验证集的组数
SEED = 42

# 模型输出分辨率与最长边相关的最小公倍数（U-Net 4 次下采样需能被 16 整除）
STRIDE = 16
