import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_l3(nn.Module):

    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet_l3, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool3d(3)
        self.fc = nn.Linear(9216 * block.expansion, 512)
        self.fc2 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)        
        x = self.layer1(x)        
        x = self.layer2(x)        
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x1 = self.fc(x)
        x = self.fc2(x1)
        return [x,x1]


class AlexNet3D_Dropout(nn.Module):

    def __init__(self, num_classes = 2):
        super(AlexNet3D_Dropout, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU( inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU( inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),

            nn.Conv3d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU( inplace=True),

            nn.Conv3d(192, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU( inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),
        )
       
        self.classifier = nn.Sequential(nn.Dropout(),
                                        nn.Linear(256, 64),
                                        nn.ReLU( inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(64, num_classes),
                                       )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
          
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return [x, x]


class AlexNet3D_Deeper_Dropout(nn.Module):

    def __init__(self, num_classes = 2):
        super(AlexNet3D_Deeper_Dropout, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU( inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU( inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),

            nn.Conv3d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm3d(384),
            nn.ReLU( inplace=True),

            nn.Conv3d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU( inplace=True),

            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU( inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),
        )
       
        self.classifier = nn.Sequential(nn.Dropout(),
                                        nn.Linear(512, 64),
                                        nn.ReLU( inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(64, num_classes),
                                       )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
          
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return [x, x]

class AlexNet3D_Dropout_Regression(nn.Module):

    def __init__(self, num_classes = 1):
        super(AlexNet3D_Dropout_Regression, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU( inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU( inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),

            nn.Conv3d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU( inplace=True),

            nn.Conv3d(192, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU( inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),
        )
       
        self.regressor = nn.Sequential(nn.Dropout(),
                                        #nn.Linear(1536, 64), ADNI
                                        nn.Linear(256, 64),
                                        nn.ReLU( inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(64, num_classes)
                                       )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
          
    def forward(self,x):
        xp = self.features(x)
        x = xp.view(xp.size(0), -1)
        x = self.regressor(x).squeeze()
        return [x, xp]