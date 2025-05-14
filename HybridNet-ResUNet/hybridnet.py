import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x + identity

# Attention Gate (without Sequential)
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, 1)
        self.bn_g = nn.BatchNorm2d(F_int)

        self.W_x = nn.Conv2d(F_l, F_int, 1)
        self.bn_x = nn.BatchNorm2d(F_int)

        self.psi = nn.Conv2d(F_int, 1, 1)
        self.bn_psi = nn.BatchNorm2d(1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        g1 = self.W_g(g)
        g1 = self.bn_g(g1)

        x1 = self.W_x(x)
        x1 = self.bn_x(x1)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        psi = self.bn_psi(psi)
        psi = self.sigmoid(psi)

        return x * psi

# Mini ASPP (without Sequential)
class MiniASPP(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 1)
        self.conv2 = nn.Conv2d(in_c, out_c, 3, padding=6, dilation=6)
        self.conv3 = nn.Conv2d(in_c, out_c, 3, padding=12, dilation=12)
        self.conv4 = nn.Conv2d(in_c, out_c, 3, padding=18, dilation=18)
        self.project = nn.Conv2d(out_c * 4, out_c, 1)
        self.dropout = nn.Dropout2d(0.3)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.project(out)
        out = self.dropout(out)
        return out

# Full SimpleHybridSegNetV2
class HybridNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, filter=32):
        super().__init__()
        f = filter

        # Encoder
        self.enc1 = ResidualDoubleConv(in_channels, f)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ResidualDoubleConv(f, f*2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ResidualDoubleConv(f*2, f*4)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = MiniASPP(f*4, f*8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(f*8, f*4, 2, stride=2)
        self.att3 = AttentionGate(f*4, f*4, f*2)
        self.dec3 = ResidualDoubleConv(f*8, f*4)

        self.up2 = nn.ConvTranspose2d(f*4, f*2, 2, stride=2)
        self.att2 = AttentionGate(f*2, f*2, f)
        self.dec2 = ResidualDoubleConv(f*4, f*2)

        self.up1 = nn.ConvTranspose2d(f*2, f, 2, stride=2)
        self.att1 = AttentionGate(f, f, f//2)
        self.dec1 = ResidualDoubleConv(f*2, f)

        self.final = nn.Conv2d(f, out_channels, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        d3 = self.up3(b)
        if d3.shape[-2:] != e3.shape[-2:]:
            e3 = F.interpolate(e3, size=d3.shape[-2:], mode="bilinear", align_corners=False)
        att3 = self.att3(d3, e3)
        d3 = torch.cat([att3, d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        if d2.shape[-2:] != e2.shape[-2:]:
            e2 = F.interpolate(e2, size=d2.shape[-2:], mode="bilinear", align_corners=False)
        att2 = self.att2(d2, e2)
        d2 = torch.cat([att2, d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            e1 = F.interpolate(e1, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        att1 = self.att1(d1, e1)
        d1 = torch.cat([att1, d1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return out
