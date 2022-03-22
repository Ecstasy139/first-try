import torch
import torchvision.models
from torch import nn


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        trained_model = torchvision.models.resnet50(pretrained=True)
        self.model = nn.Sequential(
            *list(trained_model.children())[:-10],
            # nn.Flatten(),
            # nn.Linear(2048, 10),
            nn.Conv2d(3, 5, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.model(x)
        return out


def main():
    a = torch.randn(2, 3, 224, 224)
    net = ResNet()
    out = net(a)
    print(out.shape)


if __name__ == '__main__':
    main()
