import os.path

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from model_Unet import *
from dataloader_heatmap import *


"""
Training the model based on the Unet, the output is heatmap
"""

def train():
    # Setting the hyper parameters
    epochs = 20
    batch_size = 1
    learning_rate = 0.01
    weight_path = 'params/net_unet_heatmap.pth'
    device = torch.device('cuda')
    net = UNet().to(device) # Load the model

    if os.path.exists(weight_path):  # Load the weight file
        net.load_state_dict(torch.load(weight_path))
        print("Successfully load the weight")
    else:
        print("There is no weight file")

    optim = torch.optim.Adam(params=net.parameters(), lr=learning_rate)  # Setting the Optimizer
    loss_fn = torch.nn.BCELoss().to(device)  # Setting the loss function

    data = xmldataset(root='data_center2.txt')

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        net.train()
        for i, (img, label) in enumerate(dataloader):
            img, label = img.to(device), label.to(device)
            label = label.float()
            output_img = net(img)
            loss = loss_fn(output_img, label)
            # print(output_img.shape, label.shape)
            print('{}-{}-Loss:{}'.format(epoch, i, loss))

            optim.zero_grad()
            loss.backward()
            optim.step()

        if epoch % 10 == 0:
            torch.save(net.state_dict(), f'params/net_unet_heatmap.pth')


def main():
    train()


if __name__ == '__main__':
    main()
