import jittor as jt
from jittor import nn
from jittor import Module
# from jittor import init
# from jittor.contrib import concat  #, argmax_pool
# import time

from utils.pyt_utils import load_model
from .CC import CC_module as CrissCrossAttention 

# TODO parallel
BatchNorm2d = nn.BatchNorm

class Bottleneck(Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        self.relu_inplace = nn.ReLU()  # retain original names
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def execute(self, x):
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
        out = self.relu_inplace(out)  # retain original names

        return out

class RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels // 4  # 512
        self.conva = nn.Sequential(nn.Conv(in_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm2d(inter_channels), nn.ReLU())
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm2d(inter_channels), nn.ReLU())

        self.bottleneck = nn.Sequential(  # 2048+512 = 2560, 512
            nn.Conv(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x, recurrence=1):
        output = self.conva(x)
        for _ in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)

        output = self.bottleneck(jt.concat([x, output], 1))  # debug
        return output

class ResNet(Module):
    def __init__(self, block, layers, num_classes, criterion, recurrence, output_stride=8):
        super(ResNet, self).__init__()
        
        self.inplanes = 128  # jittor-resnet self.inplanes = 64  # 
        blocks = [1, 1, 1]  # jittor-resnet blocks = [1, 2, 4] 
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        # self.conv1 = nn.Conv(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # jittor-resnet
        self.conv1 = nn.Conv(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU()  # jittor中无inplace参数
        self.conv2 = nn.Conv(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        # self.maxpool = nn.Pool(kernel_size=3, stride=2, padding=1)  # jittor-resnet
        self.maxpool = nn.Pool(kernel_size=3, stride=2, padding=1, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3])

        self.head = RCCAModule(2048, 512, num_classes)
        self.dsn = nn.Sequential(
            nn.Conv(1024, 512, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(512), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.criterion = criterion
        self.recurrence = recurrence

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion),  # 默认 affine_par=True
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation))

        return nn.Sequential(*layers)

    def execute(self, x, labels=None):

        x = self.relu1(self.bn1(self.conv1(x)))  # jittor-resnet only one layer
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)  # jittor-resnet x = argmax_pool(x, 2, 2)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_dsn = self.dsn(x)
        x = self.layer4(x)

        x = self.head(x, self.recurrence)

        outs = [x, x_dsn]
        if self.criterion is not None and labels is not None:
            return self.criterion(outs, labels)
        else:
            return outs


def Seg_Model(num_classes, criterion=None, pretrained_model=None, recurrence=0, **kwargs):
    model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes, criterion, recurrence)  # resnet50 [3,4,6,3]

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)

    return model