import torch
import torch.nn as nn


class DoubleConvReLUDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvReLUDown, self).__init__()

        self.module = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.module(x)


class DoubleConvReLUUp(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, output_activation='relu'):
        super(DoubleConvReLUUp, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels * 4

        if output_activation == 'relu':
            self.module = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.LeakyReLU(inplace=True),
                nn.PixelShuffle(2),
                nn.Conv2d(
                    in_channels=mid_channels // 4,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.LeakyReLU(inplace=True),
            )
        elif output_activation == 'none':
            self.module = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.LeakyReLU(inplace=True),
                nn.PixelShuffle(2),
                nn.Conv2d(
                    in_channels=mid_channels // 4,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                ),
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.module(x)


class UNet(nn.Module):
    def __init__(self, input_channel=3):
        super(UNet, self).__init__()

        self.down0 = DoubleConvReLUDown(in_channels=input_channel, out_channels=32)
        self.down1 = DoubleConvReLUDown(in_channels=32, out_channels=64)
        self.down2 = DoubleConvReLUDown(in_channels=64, out_channels=128)
        self.down3 = DoubleConvReLUDown(in_channels=128, out_channels=256)
        self.down4 = DoubleConvReLUDown(in_channels=256, out_channels=512)

        self.up4 = DoubleConvReLUUp(in_channels=512, out_channels=256)
        self.up3 = DoubleConvReLUUp(in_channels=256, out_channels=128)
        self.up2 = DoubleConvReLUUp(in_channels=128, out_channels=64)
        self.up1 = DoubleConvReLUUp(in_channels=128, out_channels=96)
        self.up0 = DoubleConvReLUUp(in_channels=128, out_channels=3, mid_channels=384, output_activation='none')

    def forward(self, x):
        x0 = self.down0(x)  # 32, 1/2
        x1 = self.down1(x0)  # 64, 1/4
        x2 = self.down2(x1)  # 128, 1/8
        x3 = self.down3(x2)  # 256, 1/16
        x4 = self.down4(x3)  # 512, 1/32

        x3 = self.up4(x4) + x3  # 256, 1/16
        x2 = self.up3(x3) + x2  # 128, 1/8
        x1 = torch.cat([self.up2(x2), x1], dim=1)  # 128, 1/4, densely connection
        x0 = torch.cat([self.up1(x1), x0], dim=1)  # 128, 1/2, densely connection
        x = torch.tanh(self.up0(x0) + x[:, :3, :, :])  # 3, 1/1

        return x
