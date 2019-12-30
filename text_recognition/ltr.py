import torch
import os,sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from text_recognition.backbone.Res import ResNet
from text_recognition.backbone.Res import BasicBlock, Bottleneck
import torch.nn as nn
import torch.nn.functional as F
import opt

# Res101 : Bottleneck, [3, 4, 23, 3]
# Res18 : BasicBlock, [2, 2, 2, 2]


class LTR(ResNet): #res101

    def __init__(self):
        super(LTR, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=opt.NUM_CLASSES)
        self.conv1 = torch.nn.Conv2d(1, 64,
                                     kernel_size=(7, 7),
                                     stride=(2, 2),
                                     padding=(3, 3), bias=False)

    def forward(self, x):
        return super(LTR, self).forward(x)


class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(774400, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = F.relu(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = F.max_pool2d(x, 2)
        print(x.shape)
        x = self.dropout1(x)
        print(x.shape)
        x = torch.flatten(x, 1)
        print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.softmax(x, dim=1)
        return output



