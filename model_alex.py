import torch
import torchvision
from torch import nn


class AlexNet(nn.Module):
    """
    AlexNet model based on transfer learning
    """
    def __init__(self):
        super(AlexNet, self).__init__()
        trained_model = torchvision.models.alexnet(pretrained=True)
        self.model = nn.Sequential(
            trained_model,
            nn.Flatten(),
            nn.Linear(1000, 10)
        )
        # print(trained_model)

    def forward(self, x):
        out = self.model(x)
        return out


def main():
    """
    Testing of the code by creating an element
    :return:
    """
    model = AlexNet()
    a = torch.randn(1, 3, 224, 224)
    out = model(a)
    print(out.shape)



if __name__ == '__main__':
    main()