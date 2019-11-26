import torch
from backbone.ResNet import ResNet
from backbone.ResNet import BasicBlock
import config

class WTR(ResNet): #res18

    def __init__(self):
        super(WTR, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=config.NUM_CLASSES)
        self.conv1 = torch.nn.Conv2d(1, 64,
                                     kernel_size=(7, 7),
                                     stride=(2, 2),
                                     padding=(3, 3), bias=False)

    def forward(self, x):
        return super(WTR, self).forward(x)