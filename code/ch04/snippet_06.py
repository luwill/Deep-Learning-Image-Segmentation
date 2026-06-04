# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 4 章 — (无标号片段)
# 原始位置：book-review/drafts/ch04-v3.md:509
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
class Res_UNet(nn.Module):
    def __init__(self, in_channel, out_channel, block, num_block):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=(7, 7),
                      stride=(2, 2), padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        self.dconv_last = nn.Sequential(
            nn.Conv2d(128, 64, (3, 3), padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, out_channel, (1, 1)),
        )

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        conv1 = self.conv1(x)
        temp = self.maxpool(conv1)
        conv2 = self.conv2_x(temp)
        conv3 = self.conv3_x(conv2)
        conv4 = self.conv4_x(conv3)
        bottle = self.conv5_x(conv4)

        x = self.upsample(bottle)
        x = torch.cat([x, conv4], dim=1); x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1); x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1); x = self.dconv_up1(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        return self.dconv_last(x)
