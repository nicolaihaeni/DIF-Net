import torch
from torch import nn
import torch.nn.functional as F


class DepthPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.in_dim = 7
        self.depth_predictor = StackedHourglassNet(in_dim=self.in_dim, out_dim=1)

    def forward(self, images, masks, normals):
        inputs = torch.cat([images, masks[..., None], normals], -1).permute(0, 3, 1, 2)
        depth = self.depth_predictor(inputs).squeeze(1)
        return depth


class NormalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_dim = 4
        self.normal_predictor = StackedHourglassNet(in_dim=self.in_dim, out_dim=3)

    def forward(self, images, masks):
        inputs = torch.cat([images, masks[..., None]], -1).contiguous()

        if inputs.shape[1] != self.in_dim:
            inputs = inputs.permute(0, 3, 1, 2)
        return self.normal_predictor(inputs).permute(0, 2, 3, 1)


class StackedHourglassNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        kernel_size = 3

        self.conv0 = conv(in_dim, 64, kernel_size)
        self.conv1 = conv(64, 64, kernel_size)
        self.conv2 = conv(64, 64, kernel_size)

        self.p0 = nn.MaxPool2d(2)
        self.conv3 = conv(64, 128, kernel_size)

        self.p1 = nn.MaxPool2d(2)
        self.conv4 = conv(128, 256, kernel_size)

        self.p2 = nn.MaxPool2d(2)
        self.conv5 = conv(256, 512, kernel_size)

        self.p3 = nn.MaxPool2d(2)
        self.conv6 = conv(512, 1024, kernel_size)

        self.conv7 = conv(1024, 1024, kernel_size)
        self.conv8 = conv(1024, 512, kernel_size)

        self.conv9 = conv(1024, 512, kernel_size)
        self.conv10 = conv(512, 256, kernel_size)

        self.conv11 = conv(512, 256, kernel_size)
        self.conv12 = conv(256, 128, kernel_size)

        self.conv13 = conv(256, 128, kernel_size)
        self.conv14 = conv(128, 64, kernel_size)

        self.conv15 = conv(128, 64, kernel_size)
        self.conv16 = conv(64, 64, kernel_size)

        self.out = conv(64, out_dim, 1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(self.p0(conv2))
        conv4 = self.conv4(self.p1(conv3))
        conv5 = self.conv5(self.p2(conv4))
        x = self.conv6(self.p3(conv5))
        x = self.conv7(x)
        x = self.conv8(x)

        up1 = F.interpolate(x, scale_factor=2, mode="bilinear")
        cat1 = torch.cat([up1, conv5], dim=1)

        x = self.conv9(cat1)
        x = self.conv10(x)

        up2 = F.interpolate(x, scale_factor=2, mode="bilinear")
        cat2 = torch.cat([up2, conv4], dim=1)

        x = self.conv11(cat2)
        x = self.conv12(x)

        up3 = F.interpolate(x, scale_factor=2, mode="bilinear")
        cat3 = torch.cat([up3, conv3], dim=1)

        x = self.conv13(cat3)
        x = self.conv14(x)

        up4 = F.interpolate(x, scale_factor=2, mode="bilinear")
        cat4 = torch.cat([up4, conv2], dim=1)

        x = self.conv15(cat4)
        x = self.conv16(x)
        return self.out(x)


class conv(nn.Conv2d):
    def __init__(
        self, input_dim: int, out_dim: int, *args, bn=True, relu=True, **kwargs
    ):
        super().__init__(input_dim, out_dim, padding="same", *args, **kwargs)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)
        else:
            self.bn = None
        if relu:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = None

    def forward(self, x):
        out = super().forward(x)
        if self.bn is not None:
            out = self.bn(out)
        if self.activation is not None:
            out = self.activation(out)
        return out
