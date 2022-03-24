import os
import numpy as np
import torch
from PIL import Image,ImageDraw, ImageFont
from torchvision.models import resnet18
from model_res import *
import torchvision
from torch import nn
from dataloader2 import *
from transform import *


def findlabels(img):
    """
    Find the testing images' landmarks of ground truth to compare with the predicted ones
    :param img: The testing image
    :return: The ground truth of labels of the input images
    """
    center_file = 'data_center2.txt'
    f = open(center_file, 'r')
    dataset = f.readlines()
    head_path = r'part-of-AFLW/already_labeled/'
    labels = np.zeros(10)
    labels_int = []
    img_path = os.path.join(head_path, img)

    for data in dataset:
        if img_path == data.split(' ')[0]:
            labels = data.split(' ')[1:]

    for label in labels:
        labels_int.append(int(label))
    return labels_int


def detection(img, ldmks, type):
    """
    Exhibit the landmarks (either the ground truth or the prediction) on the images
    :param img: the testing image
    :param ldmks: the landmarks of the image (either the ground truth or the prediction)
    :param type: the type of the landmarks: real (ground truth) or detection (prediction)
    :return: image with the landmarks and the text on it
    """
    draw = ImageDraw.Draw(img)
    w, h = img.size
    ldmks = np.array(ldmks)
    setFont = ImageFont.truetype('C:/windows/fonts/times.ttf', int(w / 20))
    if type == 'real':
        # if the type of the landmarks is 'real' then exhibit the landmarks on the images and write 'ground truth'
        if ldmks.all() == 0:
            # if the labels are all zero, it means the image is not from the dataset, therefore there is no ground truth
            draw.text((0, 0), "There is no Ground Truth.", fill=(0, 0, 0), font=setFont)

        else:
            draw.text((0, 0), "Ground Truth", fill=(0, 0, 0), font=setFont)
            for j in range(0, 10, 2):
                 draw.ellipse((ldmks[j] - 10, ldmks[j + 1] - 10, ldmks[j] + 10, ldmks[j + 1] + 10), (255, 0, 0))

    elif type == 'detection':
        # if the type of the landmarks is 'detection' then exhibit the landmarks on the images and write 'Prediction'
        draw.text((0, 0), "Prediction", fill=(0, 0, 0), font=setFont)
        for j in range(0, 10, 2):
             draw.ellipse((ldmks[j] - 10, ldmks[j + 1] - 10, ldmks[j] + 10, ldmks[j + 1] + 10), (255, 0, 0))


    return img


def showresults(img1, img2):
    """
    Joint prediction image and ground truth to compare
    :param img1: the image with ground truth landmarks
    :param img2: the image with predicted landmarks
    :return: the joint image of img1 and img2
    """
    w1, h1 = img1.size
    w2, h2 = img2.size
    if w1 != w2 or h1 != h2:
        # if 2 images are not equal in size, the output image will be all black
        flag = False
    else:
        # joint 2 images
        img_ = Image.new('RGB', size=(2 * w, h))
        img_.paste(img1, (0, 0, w, h))
        img_.paste(img2, (w, 0, 2 * w, h))
        flag = img_

    return flag


path='test_image'
model = ResNet()  # Load Resnet model
model.load_state_dict(torch.load('params/net_res.pth'))
data = xmldataset(root='data_center2.txt')
tot = torchvision.transforms.ToTensor()
compose = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    # torchvision.transforms.RandomCrop((224, 224)),
    # torchvision.transforms.ToTensor()
])

model.eval()
for l in os.listdir(path):
    img_detect = Image.open(os.path.join(path, l))
    img_real = Image.open(os.path.join(path, l))
    real_ldmks = findlabels(l) # get ground truth landmarks
    w, h = img_real.size
    img_data = compose(img_detect)
    draw_detect = ImageDraw.Draw(img_detect)
    draw_real = ImageDraw.Draw(img_real)
    img1 = tot(img_data)
    img1 = torch.unsqueeze(img1, dim=0)
    out_ = model(img1) # get prediction
    print(out_)
    out_ = out_[0].tolist()
    print(out_)
    k = 0
    out = []
    for i in range(0, 9, 2):
        out.append(out_[i] * w)
        out.append(out_[i+1] * h)
    print(out)
    img_real = detection(img_real, real_ldmks, 'real')
    img_detect = detection(img_detect, out, 'detection')
    img_inone = showresults(img_detect, img_real)
    img_inone.show()


