import os.path

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model_res import *
from dataloader2 import *

"""
Training the model based on the ResNet, the output is the coordinates of 5 landmarks
"""


def train():
    # Setting hyper parameters
    epochs = 40
    batch_size = 10
    learning_rate = 0.01
    total_loss = 0
    writer = SummaryWriter('logs_train')
    weight_path = 'params/net_res1.pth'
    device = torch.device('cuda')
    net = ResNet().to(device)
    if os.path.exists(weight_path):  # Load weight file
        net.load_state_dict(torch.load(weight_path))
        print("Successfully load the weight")
    else:
        print("There is no weight file")

    optim = torch.optim.Adam(params=net.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss().to(device)


    data = xmldataset(root='data_center2.txt')
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)


    for epoch in range(epochs):
        net.train()
        total_loss = 0
        for i, (img, label) in enumerate(dataloader):
            img, label = img.to(device), label.to(device)
            #print(img.shape, label)
            output = net(img)
            output = output.to(torch.float64)
            loss = loss_fn(output, label)
            # print(output.dtype, loss.dtype)
            print('{}-{}-Loss:{}'.format(epoch, i, loss))
            total_loss = total_loss + loss

            optim.zero_grad()
            loss.backward()
            optim.step()

        writer.add_scalar('train_loss', total_loss.item(), epoch)

        if epoch % 10 == 0:
            torch.save(net.state_dict(), f'params/net_res1.pth')

    writer.close()


def main():
    train()


if __name__ == '__main__':
    main()
