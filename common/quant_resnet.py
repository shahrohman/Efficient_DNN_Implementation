import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from brevitas.quant import SignedBinaryWeightPerTensorConst
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.nn import QuantReLU
from brevitas.quant import Int8ActPerTensorFloat
import brevitas.nn as qnn



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return  qnn.QuantConv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride, padding=1,bias = False,
                                     weight_quant = Int8WeightPerTensorFloat,
                                     weight_bit_width=3,
                                     input_quant=Int8ActPerTensorFloat,
                                     input_bit_width=8,
                                     #output_quant=Int8WeightPerTensorFloat,
                                     #output_bit_width=8
                                     )

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return qnn.QuantConv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, stride=stride,bias = False,
                             weight_quant =Int8WeightPerTensorFloat,
                             weight_bit_width=3,
                             input_quant=Int8ActPerTensorFloat,
                             input_bit_width=8,
                             #output_quant=Int8WeightPerTensorFloat,
                             #output_bit_width=8
                             )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu =  qnn.QuantReLU(bit_width=8, inplace = True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class CifarResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(CifarResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu =  qnn.QuantReLU(bit_width=8, inplace = True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc =  qnn.QuantLinear(64 * block.expansion,num_classes, weight_bit_width=3,bias = True)

        for m in self.modules():
            if isinstance(m, qnn.QuantConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

