""" Full assembly of the parts to form the complete network """
import torch
import torch.nn.functional as F

from .unet_parts import *


def get_unet(n_channels, n_classes, from_exp, to_exp):

    return UNetMedium(n_channels=n_channels,
                      n_classes=n_classes,
                      from_exp=from_exp,
                      to_exp=to_exp)


class UNetMedium(nn.Module):
    def __init__(self, n_channels, n_classes, from_exp, to_exp):
        # 4 mil params
        super(UNetMedium, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        bilinear = True
        self.from_exp = from_exp
        self.to_exp = to_exp
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, inp):
        x, postproc_fcn = inp
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return postproc_fcn(logits)
