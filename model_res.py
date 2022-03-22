import torch
import torchvision.models
from torch import nn


class ResNet(nn.Module):
    """
    ResNet model based on transfer learning
    """
    def __init__(self):
        super(ResNet, self).__init__()
        trained_model = torchvision.models.resnet50(pretrained=True)
        # print(trained_model)
        self.model = nn.Sequential(
            *list(trained_model.children())[:-1],
            nn.Flatten(),
            nn.Linear(2048, 10)
            # nn.Sigmoid()
        )

    def forward(self, x):
        out = self.model(x)
        return out


def main():
    """
    Testing of the model by creating an element
    :return:
    """
    a = torch.randn(2, 3, 224, 224)
    net = ResNet()
    # print(net.model)
    out = net(a)
    print(out.shape)


if __name__ == '__main__':
    main()
