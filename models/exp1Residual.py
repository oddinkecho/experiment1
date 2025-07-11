import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,downsample=False):
        super().__init__()
        stride=2 if downsample else 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class lwResNet6(nn.Module):
    def __init__(self):
        super().__init__()

        # 残差块
        self.layer1 = ResidualBlock(3, 20)
        self.layer2 = ResidualBlock(20, 40, downsample=True)
        self.layer3 = ResidualBlock(40, 60, downsample=True)
        self.layer4 = ResidualBlock(60, 80, downsample=True)
        self.layer5 = ResidualBlock(80, 100, downsample=True)
        self.layer6 = ResidualBlock(100, 120, downsample=True)

        self.fc = nn.Linear(120, 10)

    def forward(self, x):
        #(3,32,32)
        out = self.layer1(x)#(20,32,32)
        out = self.layer2(out)#(40,16,16)
        out = self.layer3(out)#(60,8,)
        out = self.layer4(out)#(80,4,)
        out = self.layer5(out)#(100,2,)
        out = self.layer6(out)#(120,1,)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out