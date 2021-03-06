import os.path
from PFLD_Loss import *
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PFLD import *
from dataloader_PFLD import *
from adabelief_pytorch import AdaBelief

"""
Training the model based on the ResNet, the output is the coordinates of 5 landmarks
"""


def train():
    # Setting hyper parameters
    epochs = 100
    batch_size = 10
    learning_rate = 0.001
    total_loss = 0
    # weight_decay = 0.000001
    writer = SummaryWriter('logs_PLFD')
    weight_path = 'params/net_PFLD.pth'
    device = torch.device('cuda')
    net = PFLDInference().to(device)
    if os.path.exists(weight_path):  # Load weight file
        net.load_state_dict(torch.load(weight_path))
        print("Successfully load the weight")
    else:
        print("There is no weight file")

    optim = AdaBelief(net.parameters(), lr=learning_rate, eps=1e-16, betas=(0.9, 0.999), weight_decouple=False, rectify=True)
    loss_fn = torch.nn.MSELoss().to(device)

    data = xmldataset(root='data_center2.txt')
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [30, 80])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', patience=10, verbose=True)

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
        scheduler.step(total_loss)

        writer.add_scalar('train_loss', total_loss.item(), epoch)

        if (epoch+1) % 10 == 0:
            torch.save(net.state_dict(), f'params/net_PFLD.pth')

    writer.close()


def main():
    train()


if __name__ == '__main__':
    main()
