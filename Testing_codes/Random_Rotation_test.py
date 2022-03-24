import torchvision
import torch
from PIL import Image
from imgaug import KeypointsOnImage, Keypoint
from transform import *
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa


img_path = r'../part-of-AFLW/already_labeled/image00014.jpg'
landmarks = [1064, 1085, 1555, 1061, 1322, 2001, 1388, 2295, 1466, 2738]
img = Image.open(img_path)
draw = ImageDraw.Draw(img)
res = Rescale((256, 256))
ran = RandomCrop((224, 224))
rot = torchvision.transforms.RandomRotation

for i in range(0,len(landmarks),2):
    draw.ellipse((landmarks[i]-20, landmarks[i+1]-20,landmarks[i]+20,landmarks[i+1]+20),(255,0,0))
img.show()

img_, landmarks_ = res(img, landmarks)
img_, landmarks_ = ran(img_, landmarks_)
print(landmarks_)
img_ = np.array(img_)
seq = iaa.Sequential([
    iaa.Affine(
        rotate=(-10, 10)
    )
])
kps = KeypointsOnImage([
    Keypoint(x=landmarks_[0], y=landmarks_[1]),
    Keypoint(x=landmarks_[2], y=landmarks_[3]),
    Keypoint(x=landmarks_[4], y=landmarks_[5]),
    Keypoint(x=landmarks_[6], y=landmarks_[7]),
    Keypoint(x=landmarks_[8], y=landmarks_[9])
], shape=img_.shape)
img_, landmarks_ = seq(image=img_,keypoints=kps)
img_= np.array(img_)
img_ = Image.fromarray(np.uint8(img_))
print(landmarks_)
landmarks_after = np.zeros(10)
landmarks_after[0] = landmarks_[0].x
landmarks_after[1] = landmarks_[0].y
landmarks_after[2] = landmarks_[1].x
landmarks_after[3] = landmarks_[1].y
landmarks_after[4] = landmarks_[2].x
landmarks_after[5] = landmarks_[2].y
landmarks_after[6] = landmarks_[3].x
landmarks_after[7] = landmarks_[3].y
landmarks_after[8] = landmarks_[4].x
landmarks_after[9] = landmarks_[4].y
print(landmarks_after)

draw_ = ImageDraw.Draw(img_)

for i in range(0,len(landmarks_after),2):
    draw_.ellipse((landmarks_after[i]-2, landmarks_after[i+1]-2,landmarks_after[i]+2,landmarks_after[i+1]+2),(255,0,0))
img_.show()