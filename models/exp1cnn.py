import numpy as np
import torch
import torchvision
import torchtext
import seaborn 
import pandas 
import matplotlib
import torch.nn as nn
import math
import torch.nn.functional as F

#print("hello py")
#print(torch.__version__)
#print(torchvision.__version__)
#print(torchtext.__version__)
#print("seaborn", seaborn.__version__)
#print("pandas", pandas.__version__)
#print("matplotlib", matplotlib.__version__)
#print("PyTorch version:", torch.__version__)
#print("CUDA available:", torch.cuda.is_available())
#print("CUDA device count:", torch.cuda.device_count())
#print("Device name:", torch.cuda.get_device_name(0))


class lwCNN(nn.Module):
    def __init__(self):
        super(lwCNN, self).__init__()
       # ---- 卷积块 1 ----
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        
        # ---- 卷积块 2 ----
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(64)
        
        # ---- 卷积块 3 ----
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5   = nn.BatchNorm2d(128)
        
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6   = nn.BatchNorm2d(128)

        # ---- 池化 ----
        self.pool = nn.MaxPool2d(2, 2)

        # ---- 全连接 ----
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)      
    
    def forward(self,x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # 第一次池化 32x32 -> 16x16

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)  # 第二次池化 16x16 -> 8x8

        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)  # 第三次池化 8x8 -> 4x4

        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x