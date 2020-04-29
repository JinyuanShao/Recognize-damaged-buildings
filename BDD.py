import timm
import torch
import torch.nn as nn


class BDD_Encoder(nn.Module):
    def __init__(self):
        m = timm.create_model("efficientnet_b0", pretrained=True)
        fuseconv = list(m.blocks[3][0].children())[0]
        fuseconv.weight = nn.Parameter(
            torch.cat([fuseconv.weight, fuseconv.weight], dim=1))

        super().__init__()
        self.block0 = nn.Sequential(m.conv_stem, m.bn1, nn.LeakyReLU(inplace=True))
        self.block1 = m.blocks[0]
        self.block2 = m.blocks[1]
        self.block3 = m.blocks[2]
        self.block4 = m.blocks[3]
        self.block5 = m.blocks[4]
        self.block6 = m.blocks[5]
        self.block7 = m.blocks[6]

    def forward(self, x): # This model expects a six channels image which is concatenated by pre- and post disaster images  
        [a, b] = x  
        a, b = self.block0(a), self.block0(b)
        stage0 = torch.cat([a, b], dim=1)
        a, b = self.block1(a), self.block1(b)
        stage1 = torch.cat([a, b], dim=1)
        a, b = self.block2(a), self.block2(b)
        stage2 = torch.cat([a, b], dim=1)
        a, b = self.block3(a), self.block3(b)

        x = stage3 = torch.cat([a, b], dim=1)
        x = stage4 = self.block4(x)
        x = stage5 = self.block5(x)
        x = stage6 = self.block6(x)
        x = stage7 = self.block7(x)

        return stage7, stage6, stage5, stage4, stage3, stage2, stage1, stage0


def ConvBlock(in_channels, out_channels, ksize=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, ksize, 1, ksize // 2),
        nn.BatchNorm2d(out_channels), 
        nn.LeakyReLU(inplace=True),
    )


class BDD(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BDD_Encoder()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.h7 = ConvBlock(320, 256)
        self.h6 = ConvBlock(256 + 192, 256)
        self.h5 = ConvBlock(256 + 112, 256)
        self.h4 = ConvBlock(256 + 80, 256)
        self.h3 = ConvBlock(256 + 80, 128)
        self.h2 = ConvBlock(128 + 48, 128)
        self.h1 = ConvBlock(128 + 32, 64)
        self.h0 = ConvBlock(64 + 64, 64)
        self.finalconv = nn.Conv2d(64, 3, 1, 1)

    def forward(self, x):
        stage7, stage6, stage5, stage4, stage3, stage2, stage1, stage0 = self.encoder(x)
        x = self.h7(stage7)
        x = self.upsample(x)
        x = self.h6(torch.cat([x, stage6.detach()], dim=1))
        x = self.upsample(x)
        x = self.h5(torch.cat([x, stage5.detach()], dim=1))
        x = self.upsample(x)
        x = self.h4(torch.cat([x, stage4.detach()], dim=1))
        x = self.upsample(x)
        x = self.h3(torch.cat([x, stage3.detach()], dim=1))
        x = self.upsample(x)
        x = self.h2(torch.cat([x, stage2.detach()], dim=1))
        x = self.upsample(x)
        x = self.h1(torch.cat([x, stage1.detach()], dim=1))
        x = self.upsample(x)
        x = self.h0(torch.cat([x, stage0.detach()], dim=1))
        x = self.finalconv(x)
        x = self.upsample(x)

        return x