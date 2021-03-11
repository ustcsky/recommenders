# coding:utf8
from .basic_module import BasicModule
from torch import nn
from torch.nn import functional as F
import torch

class UserNet(BasicModule):

    def __init__(self, num_classes=2):
        super(UserNet, self).__init__()
        self.model_name = 'UserNet'

        use_his = []
        for i,data in enumerate(self.yq_users):
            if data in self.behaviors[0]:
                use_his.append(self.behaviors[self.yq_users[i][0]][2])
        self.layer1 = self.NewsNet(use_his)


        self.uesremb = torch.cat(self.out1, self.out2)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        """
        构建layer,包含多个residual block
        """
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel))

        layers = []
        layers.append((inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(self.NewsNet(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)
