import torch.nn as nn
from torch import cat

class Residual(nn.Module):
  def __init__(self, fn):
      super().__init__()
      self.fn = fn

  def forward(self, x):
      return self.fn(x) + x

class ConvMixerBlock(nn.Module):
  def __init__(self, dim, depth=5, kernel_size=9, patch_size=7):
    super(ConvMixerBlock, self).__init__()
    self.block = nn.Sequential(
      *[nn.Sequential(
          Residual(nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(kernel_size, kernel_size), groups=dim, padding="same"),
            nn.GELU(),
            nn.BatchNorm2d(dim)
          )),
          nn.Conv2d(dim, dim, kernel_size=(1, 1)),
          nn.GELU(),
          nn.BatchNorm2d(dim)
        ) for i in range(depth)]
    )
  def forward(self, x):
      x = self.block(x)
      return x

class ConvBlock(nn.Module):
  def __init__(self, inChannels, outChannels):
    super().__init__()
    # store the convolution and RELU layers
    self.conv = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(outChannels, outChannels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outChannels)
        )
  def forward(self, x):
    # apply CONV => RELU => CONV block to the inputs and return it
    x = self.conv(x)
    return x


class UPConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(UPConvBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outChannels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.up(x)
        return x

class UNetModel1(nn.Module):
  def __init__(self, img_ch=1, output_ch=3, depth=5, kernel=7, patch=7):
    super(UNetModel1, self).__init__()

    # Encoder
    self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.Conv1 = ConvBlock(inChannels=img_ch, outChannels=32)
    self.ConvMixer1 = ConvMixerBlock(dim=32, depth=depth, kernel_size=kernel, patch_size=patch)
    self.Conv2 = ConvBlock(inChannels=32, outChannels=64)
    self.ConvMixer2 = ConvMixerBlock(dim=64, depth=depth, kernel_size=kernel, patch_size=patch)
    self.Conv3 = ConvBlock(inChannels=64, outChannels=128)
    self.ConvMixer3 = ConvMixerBlock(dim=128, depth=depth, kernel_size=kernel, patch_size=patch)
    self.Conv4 = ConvBlock(inChannels=128, outChannels=256)
    self.ConvMixer4 = ConvMixerBlock(dim=256, depth=depth, kernel_size=kernel, patch_size=patch)
    self.Conv5 = ConvBlock(inChannels=256, outChannels=512)
    self.ConvMixer5 = ConvMixerBlock(dim=512, depth=depth, kernel_size=kernel, patch_size=patch)
    # Decoder
    self.Up5 = UPConvBlock(inChannels=512, outChannels=256)
    self.Up_conv5 = ConvBlock(inChannels=256 * 2, outChannels=256)
    self.Up4 = UPConvBlock(inChannels=256, outChannels=128)
    self.Up_conv4 = ConvBlock(inChannels=128 * 2, outChannels=128)
    self.Up3 = UPConvBlock(inChannels=128, outChannels=64)
    self.Up_conv3 = ConvBlock(inChannels=64 * 2, outChannels=64)
    self.Up2 = UPConvBlock(inChannels=64, outChannels=32)
    self.Up_conv2 = ConvBlock(inChannels=32 * 2, outChannels=32)
    self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size=1, stride=1, padding=0)
  def forward(self, x):

    x1 = self.Conv1(x)
    x1 = self.ConvMixer1(x1)

    x2 = self.Maxpool(x1)
    x2 = self.Conv2(x2)
    x2 = self.ConvMixer2(x2)

    x3 = self.Maxpool(x2)
    x3 = self.Conv3(x3)
    x3 = self.ConvMixer3(x3)

    x4 = self.Maxpool(x3)
    x4 = self.Conv4(x4)
    x4 = self.ConvMixer4(x4)

    x5 = self.Maxpool(x4)
    x5 = self.Conv5(x5)
    x5 = self.ConvMixer5(x5)

    d5 = self.Up5(x5)
    d5 = cat((x4, d5), dim=1)
    d5 = self.Up_conv5(d5)

    d4 = self.Up4(d5)
    d4 = cat((x3, d4), dim=1)
    d4 = self.Up_conv4(d4)

    d3 = self.Up3(d4)
    d3 = cat((x2, d3), dim=1)
    d3 = self.Up_conv3(d3)

    d2 = self.Up2(d3)
    d2 = cat((x1, d2), dim=1)
    d2 = self.Up_conv2(d2)
    d1 = self.Conv_1x1(d2)
    return d1