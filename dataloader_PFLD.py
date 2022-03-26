import PIL.Image
import torchvision.transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, SubsetRandomSampler
from PIL import Image
import torch
import cv2.cv2 as cv
import numpy as np
from numpy import mean
from PIL import Image
from transform import *


class xmldataset(Dataset):
    """
    The rewrited Dataset Class particularly for this project
    """
    def __init__(self, root):
        """
        Initial function of this class
        :param root: The address of the data center file
        """
        super(xmldataset, self).__init__()
        f = open(root, 'r')
        self.dataset = f.readlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Explore every element in the dataset
        :param idx: The number of each case
        :return: The data, which is one image and its 10 landmarks
        """
        data = self.dataset[idx]
        img_path = data.split(' ')[0] # Get the image

        points_int = []
        points = data.split(' ')[1:] # Get 10 landmarks
        for point in points:
            points_int.append(int(point))

        img = Image.open(img_path)
        res = Rescale((128, 128))  # Resize
        rand = RandomCrop((112, 112))  # Random Crop
        randrt = RandomRotate((-10, 10))  # Random Rotate
        img, ldmks = res(img, points_int)
        img, ldmks = rand(img, ldmks)
        img,ldmks = randrt(img, ldmks)
        totensor = torchvision.transforms.ToTensor()
        img, ldmks = totensor(img), torch.tensor(ldmks)  # Turn the image and landmarks to tensor
        ldmks = ldmks / 112  # Rescale the coordinates of landmarks to the range of 0 to 1

        return img, ldmks


def main():
    """
    Testing of the codes
    :return:
    """
    zjs = xmldataset(root='data_center2.txt')
    datalo = DataLoader(zjs, batch_size=1)
    length = len(zjs)
    epochs = 20
    batch_size = 10
    # K-Cross
    k = 10
    validate_data = zjs

    # indices = list(range(length))
    #
    # train_indices, val_indices = indices[0:100] + indices[200:], indices[100:200]
    # # print(train_indices, val_indices)
    #
    # # Creating PT data samplers and loaders:
    # train_sampler = SubsetRandomSampler(train_indices)
    # valid_sampler = SubsetRandomSampler(val_indices)
    #
    # train_loader = torch.utils.data.DataLoader(zjs, batch_size=1,
    #                                            sampler=train_sampler)
    # validation_loader = torch.utils.data.DataLoader(zjs, batch_size=1,
    #                                                 sampler=valid_sampler)
    # for i, (imgs, labels) in enumerate(datalo):
    #     print('{}-{}-{}'.format(i, imgs.shape, labels.shape))

    for i, (imgs, label) in enumerate(datalo):
        print(imgs.shape, label.shape)


if __name__ == '__main__':
    main()
