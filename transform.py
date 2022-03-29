import torch
import torchvision.transforms
from imgaug import KeypointsOnImage, Keypoint
from torchvision import transforms
import numpy as np
from dataloader2 import *
from PIL import Image, ImageDraw
import scipy
import imgaug as ia
import imgaug.augmenters as iaa


class Rescale(object):
    """
    Input: orignal image and landmarks

    Resize the image to match the input size of the model
    Modify the coordinates of the landmarks according to the change of the image

    :returns: the resized image and the landmarks
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image, landmarks):
        img, ldmks = image, landmarks

        w, h = img.size
        # print(img.size)
        # int
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        # tuple
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        trans = transforms.Resize((new_h, new_w))
        # img = Image.fromarray(img)
        img = trans(img)
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        for i in range(0, 9, 2):
            ldmks[i] = int(ldmks[i] * new_w / w)
            ldmks[i+1] = int(ldmks[i+1] * new_h / h)
        # ldmks = ldmks * [new_w / w, new_h / h]

        return img, ldmks


class RandomCrop(object):
    """
    Input: Resized image and landmarks

    Crop the image randomly for the purpose of Data Augment
    Modify the coordinates of the landmarks according to the change of the image

    :returns: Crop the Image and Landmarks
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image, landmarks):
        img, ldmks = image, landmarks
        w, h = image.size
        img = np.array(img)
        new_h, new_w = self.output_size


        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img = img[top: top + new_h,
                left: left + new_w]

        img = Image.fromarray(img)
        for i in range(0, 10, 2):
            ldmks[i] = ldmks[i] - left
            ldmks[i+1] = ldmks[i+1] - top

        ldmks = np.array(ldmks)

        return img, ldmks


class ToTensor(object):
    """
    Turn the image and landmarks to Tensor
    """
    def __call__(self, image, landmarks):
        img, ldmks = image, landmarks
        img = img.transpose((2, 0, 1))
        return Image.fromarray(img), landmarks


class RandomRotate(object):
    """
    Input: Image and the Landmarks

    Rotate the Images randomly for the purpose of Data Augment
    Modify the coordinates of the landmarks according to the change of the image

    :returns: the Image and the landmarks
    """
    def __init__(self, rotate_angle):
        assert isinstance(rotate_angle, (int, tuple))
        if isinstance(rotate_angle, int):
            self.output_size = (rotate_angle, rotate_angle)
        else:
            assert len(rotate_angle) == 2
            self.rotate_angle = rotate_angle

    def __call__(self, image, landmarks):
        seq = iaa.Sequential([
            iaa.Affine(
                rotate=self.rotate_angle
            )
        ])
        image = np.array(image)
        kps = KeypointsOnImage([
            Keypoint(x=landmarks[0], y=landmarks[1]),
            Keypoint(x=landmarks[2], y=landmarks[3]),
            Keypoint(x=landmarks[4], y=landmarks[5]),
            Keypoint(x=landmarks[6], y=landmarks[7]),
            Keypoint(x=landmarks[8], y=landmarks[9])
        ], shape=image.shape)
        img_, landmarks_ = seq(image=image, keypoints=kps)
        img_ = np.array(img_)
        img_ = Image.fromarray(np.uint8(img_))
        landmarks_after = np.zeros(10)
        landmarks_after[0] = int(landmarks_.keypoints[0].x)
        landmarks_after[1] = int(landmarks_.keypoints[0].y)
        landmarks_after[2] = int(landmarks_.keypoints[1].x)
        landmarks_after[3] = int(landmarks_.keypoints[1].y)
        landmarks_after[4] = int(landmarks_.keypoints[2].x)
        landmarks_after[5] = int(landmarks_.keypoints[2].y)
        landmarks_after[6] = int(landmarks_.keypoints[3].x)
        landmarks_after[7] = int(landmarks_.keypoints[3].y)
        landmarks_after[8] = int(landmarks_.keypoints[4].x)
        landmarks_after[9] = int(landmarks_.keypoints[4].y)

        return img_, landmarks_after


def main():
    """
    Testing of the codes
    """
    image = Image.open(r'part-of-AFLW/already_labeled/image00014.jpg')
    landmarks = [1064, 1085, 1555, 1061, 1322, 2001, 1388, 2295, 1466, 2738]
    # draw2 = ImageDraw.Draw(image)
    # for j in range(0, 10, 2):
    #     draw2.ellipse((landmarks[j] - 10, landmarks[j + 1] - 10, landmarks[j] + 10, landmarks[j + 1] + 10), (255, 0, 0))
    #
    # image.show()
    res = Rescale((256, 256))
    rand = RandomCrop((224, 224))
    randrt = RandomRotate((-20, 20))
    img, ldmks = res(image, landmarks)
    img, ldmks = rand(img, landmarks)
    img, ldmks = randrt(img, landmarks)
    # totensor = torchvision.transforms.ToTensor()
    ldmks = ldmks.astype(int)
    print(img)
    # img, ldmks = totensor(img), torch.tensor(ldmks)
    print(img, ldmks)


    print(img.size)
    draw1 = ImageDraw.Draw(img)

    for j in range(0, 10, 2):
        draw1.ellipse((ldmks[j] - 10, ldmks[j+1] - 10, ldmks[j] + 10, ldmks[j+1] + 10),(255, 0, 0))

    img.show()



if __name__ == '__main__':
    main()