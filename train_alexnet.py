import os.path

import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from model_alex import *
from dataloader2 import *

"""
Training the model based on AlexNet, the output is coordinates of 5 points
"""


def train():
    # Setting hyper parameters
    epochs = 40
    batch_size = 10
    learning_rate = 0.01

    writer = SummaryWriter('logs_train')
    weight_path = 'params/net_alex.pth'
    device = torch.device('cuda')
    net = AlexNet().to(device)  # Load the model

    if os.path.exists(weight_path):  # Load weight file
        net.load_state_dict(torch.load(weight_path))
        print("Successfully load the weight")
    else:
        print("There is no weight file")

    optim = torch.optim.Adam(params=net.parameters(), lr=learning_rate)  # Setting Optimizer
    loss_fn = torch.nn.MSELoss().to(device)  # Setting the loss function

    data = xmldataset(root='data_center2.txt')
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        net.train()
        for i, (img, label) in enumerate(dataloader):
            img, label = img.to(device), label.to(device)
            #print(img.shape, label)
            output = net(img)
            output = output.to(torch.float64)
            loss = loss_fn(output, label)
            # print(output.dtype, loss.dtype)
            writer.add_scalar('train_loss', loss.item(), i)
            print('{}-{}-Loss:{}'.format(epoch, i, loss))

            optim.zero_grad()
            loss.backward()
            optim.step()

        if epoch % 10 == 0:
            torch.save(net.state_dict(), f'params/net_alex.pth')


def main():
    train()


if __name__ == '__main__':
    main()