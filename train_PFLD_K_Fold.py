import os.path

import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from PFLD import *
from dataloader_PFLD import *
from adabelief_pytorch import AdaBelief

"""
Training the model based on ResNet, the output is the coordinates of 5 landmarks
* K-Fold Validation Included
"""


def set_lr(epoch, lr):
    if epoch in range(0, 4):
        lr_ = lr

    elif epoch in range(4, 8):
        lr_ = 0.001

    elif epoch in range(8, 12):
        lr_ = 0.0005

    elif epoch in range(12, 14):
        lr_ = 0.00025

    elif epoch in range(14, 16):
        lr_ = 0.000125

    else:
        lr_ = 0.00001

    return lr_


def train():
    # Setting hyper parameters
    epochs = 18
    batch_size = 20
    learning_rate = 2e-3
    lr = learning_rate
    weight_decay = 1e-2
    weight_path = 'params/net_PFLD_k_fold.pth'
    device = torch.device('cuda')
    net = PFLDInference().to(device)
    k = 10  # Setting the Fold
    writer = SummaryWriter('logs_PFLD_k_fold')

    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print("Successfully load the weight")
    else:
        print("There is no weight file")
    optim = AdaBelief(net.parameters(), lr=learning_rate, eps=1e-16, weight_decay=weight_decay, betas=(0.9, 0.999), weight_decouple=True, rectify=False)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', patience=10, verbose=True, min_lr=1e-6, eps=1e-16)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=40, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[80, 160], gamma=0.1)
    loss_fn = torch.nn.MSELoss().to(device)  # Setting Loss Function

    data = xmldataset(root='data_center2.txt')
    length = len(data)

    for epoch in range(epochs):
        total_loss = []
        lr_ = set_lr(epoch, lr)
        optim.param_groups[0]["lr"] = lr_
        for fold in range(k):  # Control the K Fold Loop

            train_indices, val_indices = get_k_fold(k, fold, length)

            train_sampler = SubsetRandomSampler(train_indices)  # Call the training data
            valid_sampler = SubsetRandomSampler(val_indices)  # Call the evaluating data

            train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler)
            validation_loader = DataLoader(data, batch_size=batch_size, sampler=valid_sampler)
            net.train()
            train_loss = 0

            for i, (img, label) in enumerate(train_loader):
                img, label = img.to(device), label.to(device)
                # print(img.shape, label)
                output = net(img)
                output = output.to(torch.float64)
                loss = loss_fn(output, label)
                print('Training: epoch: {}, Validate Fold: {}, Batch: {}, Loss: {}, Learning rate:{}'.format(epoch, fold, i, loss, optim.param_groups[0]["lr"]))
                train_loss = train_loss + loss

                optim.zero_grad()
                loss.backward()
                optim.step()


            writer.add_scalar('train_loss', train_loss.item(), epoch * 10 + fold + 1)

            net.eval()
            eval_loss = 0
            with torch.no_grad():
                for i, (img, label) in enumerate(validation_loader):
                    img, label = img.to(device), label.to(device)
                    # print(img.shape, label)
                    output = net(img)
                    output = output.to(torch.float64)
                    loss = loss_fn(output, label)
                    eval_loss = eval_loss + loss
                    print('Validating: epoch: {}, Validate Fold: {}, Batch: {} Loss: {}'.format(epoch, fold, i, loss))

                print('num: {}, eval_loss: {} '.format(epoch * 10 + fold + 1, eval_loss))
                writer.add_scalar('eval_loss', eval_loss.item(), epoch * 10 + fold + 1)
            # scheduler.step()
            total_loss.append(eval_loss)



        # total_loss = np.array(total_loss)
        # loss_of_10 = total_loss.sum() / 10
        # print(loss_of_10)
        # writer.add_scalar('Average Eval Loss', loss_of_10, epoch + 1)
        # print('Epoch {} average eval loss: {}'.format(epoch, loss_of_10))

        if (epoch+1) % 1 == 0:
            torch.save(net.state_dict(), f'params/net_PFLD_k_fold.pth')
            print('Save successfully')

    writer.close()


def get_k_fold(k, fold, length):
    """
    Return the number of the fold that used to evaluate
    :param k: The number of the fold
    :param fold: how many folds
    :param length: The length of the dataset
    :return: number of the fold used to evaluate
    """
    fold_size = int(length / k)
    start = fold * fold_size
    end = (fold + 1) * fold_size
    # print(start, end)
    indices = list(range(length))
    train_indices = indices[0:start] + indices[end:]
    validate_indices = indices[start:end]
    return train_indices, validate_indices


def main():
    train()


if __name__ == '__main__':
    main()
