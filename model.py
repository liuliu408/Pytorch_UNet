import torch.nn as nn
import torch


# 搭建unet 网络
class DoubleConv(nn.Module):    # 连续两次卷积
    def __init__(self,in_channels,out_channels):
        super(DoubleConv,self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,bias=False),     # 3*3 卷积核
            nn.BatchNorm2d(out_channels),                                               # 用 BN 代替 Dropout
            nn.ReLU(inplace=True),                                                      # ReLU 激活函数

            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):    # 前向传播
        x = self.double_conv(x)
        return x


class Down(nn.Module):   # 下采样
    def __init__(self,in_channels,out_channels):
        super(Down, self).__init__()
        self.downsampling = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2),
            DoubleConv(in_channels,out_channels)
        )

    def forward(self,x):
        x = self.downsampling(x)
        return x


class Up(nn.Module):    # 上采样
    def __init__(self, in_channels, out_channels):
        super(Up,self).__init__()

        self.upsampling = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2) # 转置卷积
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.upsampling(x1)
        x = torch.cat([x2, x1], dim=1)  # 从channel 通道拼接
        x = self.conv(x)
        return x


class OutConv(nn.Module):   # 最后一个网络的输出
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):   # unet 网络
    def __init__(self, in_channels = 1, num_classes = 1):
        super(UNet, self).__init__()
        self.in_channels = in_channels                  # 输入图像的channel
        self.num_classes = num_classes                  # 网络最后的输出

        self.in_conv = DoubleConv(in_channels, 64)      # 第一层

        self.down1 = Down(64, 128)                      # 下采样过程
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024, 512)                        # 上采样过程
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.out_conv = OutConv(64, num_classes)        # 网络输出

    def forward(self, x):           # 前向传播    输入size为 (10,1,480,480)，这里设置batch = 10

        x1 = self.in_conv(x)        # torch.Size([10, 64, 480, 480])
        x2 = self.down1(x1)         # torch.Size([10, 128, 240, 240])
        x3 = self.down2(x2)         # torch.Size([10, 256, 120, 120])
        x4 = self.down3(x3)         # torch.Size([10, 512, 60, 60])
        x5 = self.down4(x4)         # torch.Size([10, 1024, 30, 30])

        x = self.up1(x5, x4)        # torch.Size([10, 512, 60, 60])
        x = self.up2(x, x3)         # torch.Size([10, 256, 120, 120])
        x = self.up3(x, x2)         # torch.Size([10, 128, 240, 240])
        x = self.up4(x, x1)         # torch.Size([10, 64, 480, 480])
        x = self.out_conv(x)        # torch.Size([10, 1, 480, 480])

        return x


# 计算 UNet 的网络参数个数
# model = UNet(in_channels=1,num_classes=1)
# print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
