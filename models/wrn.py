import torch
import torch.nn as nn
from torch.nn import init
from .layers import *
from .param_bank import *

class Block(nn.Module):
    def __init__(self, in_planes, out_planes, stride, bank=None):
        super(Block, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(in_planes)
        if bank: self.conv1 = SConv2d(bank, in_planes, out_planes, kernel_size=3, stride=stride, padding=1)
        else: self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_planes)
        if bank: self.conv2 = SConv2d(bank, out_planes, out_planes, kernel_size=3, stride=1, padding=1)
        else: self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = None
        if not self.equalInOut:
            if bank: self.convShortcut = SConv2d(bank, in_planes, out_planes, kernel_size=1, stride=stride, padding=0)
            else: self.convShortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(x))
        if not self.equalInOut: residual = out
        out = self.conv2(self.relu(self.bn2(self.conv1(out))))
        if self.convShortcut is not None: residual = self.convShortcut(residual)
        return out + residual

class WRN(nn.Module):
    def __init__(self, share_type, upsample_type, upsample_window, depth, width, num_templates, max_params, num_classes, groups):
        super(WRN, self).__init__()

        n_channels = [16, 16*width, 32*width, 64*width]
        assert((depth - 4) % 6 == 0)
        num_blocks = (depth - 4) // 6
        layers_per_bank = 2*(num_blocks-1)
        print ('WRN : Depth : {} , Widen Factor : {}, Templates per Group : {}'.format(depth, width, num_templates))

        self.num_classes = num_classes
        self.num_templates = num_templates
        self.bank = None
        if share_type != 'none':
            self.bank = ParameterGroups(groups, share_type, upsample_type, upsample_window, max_params, num_templates)

        if self.bank: self.conv_3x3 = SConv2d(self.bank, 3, n_channels[0], kernel_size=3, stride=1, padding=1)
        else: self.conv_3x3 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.stage_1 = self._make_layer(n_channels[0], n_channels[1], num_blocks, 1)
        self.stage_2 = self._make_layer(n_channels[1], n_channels[2], num_blocks, 2)
        self.stage_3 = self._make_layer(n_channels[2], n_channels[3], num_blocks, 2)

        self.lastact = nn.Sequential(nn.BatchNorm2d(n_channels[3]), nn.ReLU(inplace=True))
        self.avgpool = nn.AvgPool2d(8)
        if self.bank: self.classifier = SLinear(self.bank, n_channels[3], num_classes)
        else: self.classifier = nn.Linear(n_channels[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

        if self.bank:
            self.bank.setup_bank()

    def _make_layer(self, in_planes, out_planes, num_blocks, stride=1):
        blocks = []
        blocks.append(Block(in_planes, out_planes, stride, self.bank))
        for i in range(1, num_blocks): blocks.append(Block(out_planes, out_planes, 1, self.bank))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv_3x3(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.lastact(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def wrn(share_type, upsample_type, upsample_window, depth, width, num_templates, max_params, num_classes=10, groups=None):
    model = WRN(share_type, upsample_type, upsample_window, depth, width, num_templates, max_params, num_classes, groups)
    return model


# ImageNet model.
# This is a ResNet v1.5-style model (stride 2 on 3x3 convolutions).
# In contrast to the above, this applies batchnorm/relu after convolution.

class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 relu=True, bank=None):
        super().__init__()
        self.bank = bank
        if self.bank:
            self.conv = SConv2d(self.bank, in_channels, out_channels,
                                kernel_size=kernel_size, stride=stride,
                                padding=padding)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride, padding=padding,
                                  bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        out = self.bn(self.conv(x))
        if self.relu is not None:
            out = self.relu(out)
        return out


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, downsample, bank=None,
                 width=1, pool_residual=False):
        super().__init__()
        self.bank = bank
        self.out_channels = 4 * mid_channels
        # Width factor applies only to inner 3x3 convolution.
        mid_channels = int(mid_channels * width)

        # Skip connection.
        if downsample:
            if pool_residual:
                pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
                conv = ConvBNRelu(
                    in_channels, self.out_channels, kernel_size=1,
                    stride=1, padding=0, relu=False, bank=self.bank)
                self.skip_connection = nn.Sequential(pool, conv)
            else:
                self.skip_connection = ConvBNRelu(
                    in_channels, self.out_channels, kernel_size=1,
                    stride=2, padding=0, relu=False, bank=self.bank)
        elif in_channels != self.out_channels:
            self.skip_connection = ConvBNRelu(
                in_channels, self.out_channels, kernel_size=1,
                stride=1, padding=0, relu=False, bank=self.bank)
        else:
            self.skip_connection = None

        # Main branch.
        self.in_conv = ConvBNRelu(
            in_channels, mid_channels, kernel_size=1,
            stride=1, padding=0, bank=self.bank)
        self.mid_conv = ConvBNRelu(
            mid_channels, mid_channels, kernel_size=3,
            stride=(2 if downsample else 1), padding=1, bank=self.bank)
        self.out_conv = ConvBNRelu(
            mid_channels, self.out_channels, kernel_size=1,
            stride=1, padding=0, relu=False, bank=self.bank)
        self.out_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.skip_connection is not None:
            residual = self.skip_connection(x)
        else:
            residual = x

        out = self.out_conv(self.mid_conv(self.in_conv(x)))
        out += residual
        return self.out_relu(out)


class ResNet(nn.Module):
    def __init__(self, block, module_sizes, module_channels, num_classes,
                 width=1, bank=None, pool_residual=False):
        super().__init__()
        self.bank = bank

        # Input trunk, Inception-style.
        self.conv1 = ConvBNRelu(3, module_channels[0] // 2, kernel_size=3,
                                stride=2, padding=1, bank=self.bank)
        self.conv2 = ConvBNRelu(module_channels[0] // 2, module_channels[0] // 2,
                                kernel_size=3, stride=1, padding=1, bank=self.bank)
        self.conv3 = ConvBNRelu(module_channels[0] // 2, module_channels[0],
                                kernel_size=3, stride=1, padding=1, bank=self.bank)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Build the main network.
        modules = []
        out_channels = module_channels[0]
        for module_idx, (num_layers, mid_channels) in enumerate(zip(
                module_sizes, module_channels)):
            blocks = []
            for i in range(num_layers):
                in_channels = out_channels
                downsample = i == 0 and module_idx > 0
                b = block(in_channels, mid_channels, downsample,
                          bank=self.bank, width=width,
                          pool_residual=pool_residual)
                out_channels = b.out_channels
                blocks.append(b)
            modules.append(nn.Sequential(*blocks))
        self.block_modules = nn.Sequential(*modules)

        # Output.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if self.bank:
            self.fc = SLinear(self.bank, out_channels, num_classes)
        else:
            self.fc = nn.Linear(out_channels, num_classes)

        self._init_weights()

        if self.bank:
            self.bank.setup_bank()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                init.constant_(m.bias, 0)
        # Zero initialize the last batchnorm in each residual branch.
        for m in self.modules():
            if isinstance(m, BottleneckBlock):
                init.constant_(m.out_conv.bn.weight, 0)

    def forward(self, x):
        x = self.maxpool(self.conv3(self.conv2(self.conv1(x))))
        x = self.block_modules(x)
        x = self.fc(torch.flatten(self.avgpool(x), 1))
        return x


def wrn_imagenet(share_type, upsample_type, upsample_window, depth, width,
                  num_templates, max_params, num_classes=1000, groups=None):
    """ResNet-50, with optional width (depth ignored for now, can generalize)."""
    bank = None
    if share_type != 'none':
        bank = ParameterGroups(groups, share_type, upsample_type,
                               upsample_window, max_params, num_templates)
    return ResNet(BottleneckBlock,
                  (3, 4, 6, 3), (64, 128, 256, 512),
                  num_classes, width=width, bank=bank,
                  pool_residual=False)


def wrn_imagenet_reduced(share_type, upsample_type, upsample_window, depth,
                          width, num_templates, max_params, num_classes=1000,
                          groups=None):
    """ResNet-50 with reduced numbers of filters."""
    bank = None
    if share_type != 'none':
        bank = ParameterGroups(groups, share_type, upsample_type,
                               upsample_window, max_params, num_templates)
    return ResNet(BottleneckBlock,
                  (3, 4, 6, 3), (16, 32, 64, 96),
                  num_classes, width=width, bank=bank,
                  pool_residual=False)


def wrn_imagenet17(share_type, upsample_type, upsample_window, depth, width,
                    num_templates, max_params, num_classes=1000, groups=None):
    """Funny Wide ResNet-17 with bottleneck blocks."""
    bank = None
    if share_type != 'none':
        bank = ParameterGroups(groups, share_type, upsample_type,
                               upsample_window, max_params, num_templates)
    return ResNet(BottleneckBlock,
                  (1, 1, 1, 1), (64, 128, 256, 512),
                  num_classes, width=width, bank=bank,
                  pool_residual=False)
