import math

from torch import nn as nn

from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


class AlexNet(nn.Module):
    def __init__(self, in_channels=1, conv_channels=64):
        super().__init__()

        self.c1 = nn.Conv3d(in_channels=in_channels, out_channels=conv_channels, kernel_size=(5, 5, 5), stride=(2, 2, 2), padding=0)
        nn.BatchNorm3d(64),
        self.relu1 = nn.ReLU(inplace=True)
        self.s2 = nn.MaxPool3d(kernel_size=3, stride=3)

        self.c3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 0, 0))
        self.relu2 = nn.ReLU(inplace=True)
        self.s4 = nn.MaxPool3d(kernel_size=3, stride=3, padding=(0, 0, 0))

        self.c5 = nn.Conv3d(in_channels=128, out_channels=192, kernel_size=(3, 3, 3), padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.c6 = nn.Conv3d(in_channels=192, out_channels=192, kernel_size=(3, 3, 3), padding=1)
        self.relu4 = nn.ReLU(inplace=True)

        self.c7 = nn.Conv3d(in_channels=192, out_channels=128, kernel_size=(3, 3, 3), padding=1)
        self.relu5 = nn.ReLU(inplace=True)

        self.f8 = nn.Linear(128, 128)
        self.f9 = nn.Linear(128, 128)

        self.head_linear = nn.Linear(128, 1)

        self._init_weights()

    # see also https://github.com/pytorch/pytorch/issues/18182
    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)



    def forward(self, input_batch):
        block_out = self.c1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.s2(block_out)
        block_out = self.c3(block_out)
        block_out = self.relu2(block_out)
        block_out = self.s4(block_out)
        block_out = self.c5(block_out)
        block_out = self.relu3(block_out)
        block_out = self.c6(block_out)
        block_out = self.relu4(block_out)
        block_out = self.c7(block_out)
        block_out = self.relu5(block_out)

        conv_flat = block_out.view(
            block_out.size(0),
            -1,
        )
        linear_output = self.f8(conv_flat)
        linear_output = self.f9(linear_output)
        linear_output = self.head_linear(conv_flat)

        return linear_output


if __name__ == '__main__':
    model = AlexNet()
    print(model)
