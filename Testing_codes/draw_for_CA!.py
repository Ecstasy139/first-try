import numpy as np

from transform import *
from PIL import Image
import torch
import torchvision


path = r'D:\TorchProject\zjs_own_training\first-try\test_image1\29_0_0_20170104165154897.jpg'
landmarks = torch.randn(10)
res = Rescale((256, 256))
crop = RandomCrop((224, 224))

img = Image.open(path)

img_resize, landmarks = res(img, landmarks)
img_resize = np.array(img_resize)
img_resize = Image.fromarray(img_resize)

img_crop, landmarks = crop(img_resize, landmarks)
img_crop = np.array(img_crop)
img_crop = Image.fromarray(img_crop)

img.show()
img_resize.show()
img_crop.show()
