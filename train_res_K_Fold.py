import os.path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from model_res import *
from dataloader2 import *


"""
Training the model based on ResNet, the output is the coordinates of 5 landmarks
* K-Fold Validation Included
"""

def train():
    # Setting hyper parameters
    epochs = 40
    batch_size = 10
    learning_rate = 0.01
    weight_path = 'params/net_res_k_fold.pth'
    device = torch.device('cuda')
    net = ResNet().to(device)
    k = 10  # Setting the Fold
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print("Successfully load the weight")
    else:
        print("There is no weight file")

    optim = torch.optim.Adam(params=net.parameters(), lr=learning_rate)  # Setting the Optimizer
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)  # Setting the Learning Rate Scheduler
    loss_fn = torch.nn.MSELoss().to(device)  # Setting Loss Function

    data = xmldataset(root='data_center2.txt')
    length = len(data)

    for epoch in range(epochs):
        total_loss = []
        for fold in range(k):  # Control the K Fold Loop

            train_indices, val_indices = get_k_fold(k, fold, length)

            train_sampler = SubsetRandomSampler(train_indices)  # Call the training data
            valid_sampler = SubsetRandomSampler(val_indices)  # Call the evaluating data

            train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler)
            validation_loader = DataLoader(data, batch_size=batch_size, sampler=valid_sampler)
            net.train()
            for i, (img, label) in enumerate(train_loader):
                img, label = img.to(device), label.to(device)
                # print(img.shape, label)
                output = net(img)
                output = output.to(torch.float64)
                loss = loss_fn(output, label)
                print('Training: epoch: {}, Validate Fold: {}, Loss: {}'.format(epoch, fold, loss))

                optim.zero_grad()
                loss.backward()
                optim.step()
            scheduler.step()

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
                    print('Validating: epoch: {}, Validate Fold: {}, Loss: {}'.format(epoch, fold, eval_loss))
            total_loss.append(eval_loss)

        total_loss = np.array(total_loss)
        loss_of_10 = total_loss.sum() / 10
        print(loss_of_10)
        print('Epoch {} total loss: {}'.format(epoch, loss_of_10))

        if (epoch+1) % 10 == 0:
            torch.save(net.state_dict(), f'params/net_res_k_fold.pth')
            print('Save successfully')


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
