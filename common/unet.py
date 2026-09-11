"""经典 U-Net（含 BatchNorm）。

编码器 4 层(64→128→256→512) + 瓶颈(1024)，解码器对称；每层 DoubleConv。
输入输出长宽需为 16 的倍数（common.image_io.target_size 保证）；
因池化取整可能出现的奇偶尺寸错位，拼接前对 skip 特征做中心裁剪对齐。
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """返回 (pooled, feat)：pooled 送入下一层编码器；feat(池化前) 作为解码器跳跃连接。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        feat = self.conv(x)
        return self.pool(feat), feat


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # 中心裁剪 skip 至与 x 同尺寸（仅奇偶错位时生效）
        if x.shape[-2:] != skip.shape[-2:]:
            dh, dw = skip.shape[2] - x.shape[2], skip.shape[3] - x.shape[3]
            skip = skip[:, :, dh // 2: dh // 2 + x.shape[2],
                        dw // 2: dw // 2 + x.shape[3]]
        return self.conv(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    """in_ch 输入通道(3: RGB)；out_ch 输出通道(1: 根系前景)。"""

    def __init__(self, in_ch: int = 3, out_ch: int = 1, base: int = 64):
        super().__init__()
        features = [base, base * 2, base * 4, base * 8]  # 64,128,256,512
        self.down1 = Down(in_ch, features[0])
        self.down2 = Down(features[0], features[1])
        self.down3 = Down(features[1], features[2])
        self.down4 = Down(features[2], features[3])
        self.bottleneck = DoubleConv(features[3], features[3] * 2)      # 1024

        self.up1 = Up(features[3] * 2, features[3], features[3])
        self.up2 = Up(features[3], features[2], features[2])
        self.up3 = Up(features[2], features[1], features[1])
        self.up4 = Up(features[1], features[0], features[0])
        self.out = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x):
        p1, s1 = self.down1(x)      # skip 特征 [64]（池化前，全分辨率）
        p2, s2 = self.down2(p1)
        p3, s3 = self.down3(p2)
        p4, s4 = self.down4(p3)
        x = self.bottleneck(p4)
        x = self.up1(x, s4)
        x = self.up2(x, s3)
        x = self.up3(x, s2)
        x = self.up4(x, s1)
        return self.out(x)
