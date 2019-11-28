import torch
from backbone.ResNet import ResNet
from backbone.ResNet import BasicBlock, Bottleneck
import config


#Res101 : Bottleneck, [3, 4, 23, 3]
#Res18 : BasicBlock, [2, 2, 2, 2]

class WTR(ResNet): #res101

    def __init__(self):
        super(WTR, self).__init__(Bottleneck, [3, 4, 23, 3], num_classes=config.NUM_CLASSES)
        self.conv1 = torch.nn.Conv2d(1, 64,
                                     kernel_size=(7, 7),
                                     stride=(2, 2),
                                     padding=(3, 3), bias=False)

    def forward(self, x):
        return super(WTR, self).forward(x)
