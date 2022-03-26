import torch
from torch import nn


def conv_bn(inp, oup, kernel, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(oup), nn.ReLU(inplace=True))



class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = use_res_connect

        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio,
                      inp * expand_ratio,
                      3,
                      stride,
                      1,
                      groups=inp * expand_ratio,
                      bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)



def backbone(x):
    conv1 = nn.Conv2d(3,
                           64,
                           kernel_size=3,
                           stride=2,
                           padding=1,
                           bias=False)
    bn1 = nn.BatchNorm2d(64)
    relu = nn.ReLU(inplace=True)

    conv2 = nn.Conv2d(64,
                           64,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           bias=False)
    bn2 = nn.BatchNorm2d(64)
    relu = nn.ReLU(inplace=True)

    conv3_1 = InvertedResidual(64, 64, 2, False, 2)

    block3_2 = InvertedResidual(64, 64, 1, True, 2)
    block3_3 = InvertedResidual(64, 64, 1, True, 2)
    block3_4 = InvertedResidual(64, 64, 1, True, 2)
    block3_5 = InvertedResidual(64, 64, 1, True, 2)

    conv4_1 = InvertedResidual(64, 128, 2, False, 2)

    conv5_1 = InvertedResidual(128, 128, 1, False, 4)
    block5_2 = InvertedResidual(128, 128, 1, True, 4)
    block5_3 = InvertedResidual(128, 128, 1, True, 4)
    block5_4 = InvertedResidual(128, 128, 1, True, 4)
    block5_5 = InvertedResidual(128, 128, 1, True, 4)
    block5_6 = InvertedResidual(128, 128, 1, True, 4)

    conv6_1 = InvertedResidual(128, 16, 1, False, 2)  # [16, 14, 14]

    conv7 = conv_bn(16, 32, 3, 2)  # [32, 7, 7]
    conv8 = nn.Conv2d(32, 128, 7, 1, 0)  # [128, 1, 1]
    bn8 = nn.BatchNorm2d(128)

    avg_pool1 = nn.AvgPool2d(14)
    avg_pool2 = nn.AvgPool2d(7)
    fc = nn.Linear(176, 196)

    x = relu(bn1(conv1(x)))  # [64, 56, 56]
    x = relu(bn2(conv2(x)))  # [64, 56, 56]
    x = conv3_1(x)
    x = block3_2(x)
    x = block3_3(x)
    x = block3_4(x)
    out1 = block3_5(x)

    x = conv4_1(out1)
    x = conv5_1(x)
    x = block5_2(x)
    x = block5_3(x)
    x = block5_4(x)
    x = block5_5(x)
    x = block5_6(x)     # [1, 128, 14, 14]
    x = conv6_1(x)
    x1 = avg_pool1(x)
    x1 = x1.view(x1.size(0), -1)  # 16

    x = conv7(x)
    x2 = avg_pool2(x)
    x2 = x2.view(x2.size(0), -1)  # 32

    x3 = relu(conv8(x))
    x3 = x3.view(x3.size(0), -1)  # 128
    #
    # multi_scale = torch.cat([x1, x2, x3], 1)
    # landmarks = fc(multi_scale)

    return x1, x2, x3

def main():
    a = torch.randn(1, 3, 112, 112)
    out1, out2, out3 = backbone(a)
    print(out1.shape, out2.shape, out3.shape)


if __name__ == '__main__':
    main()
