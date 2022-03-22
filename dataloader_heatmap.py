import PIL.Image
import torchvision.transforms
from coordinate2heatmap import *
from transform import *


class xmldataset(Dataset):
    def __init__(self, root):
        super(xmldataset, self).__init__()
        f = open(root, 'r')
        self.dataset = f.readlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img_path = data.split(' ')[0]

        points_int = []
        points = data.split(' ')[1:]
        for point in points:
            points_int.append(int(point))
        img = Image.open(img_path)
        res = Rescale((256, 256))
        rand = RandomCrop((224, 224))
        rotate = RandomRotate((-10, 10))
        img, ldmks = res(img, points_int)
        img, ldmks = rand(img, ldmks)
        img, ldmks = rotate(img, ldmks)
        img = np.array(img)
        ldmks_heatmap = get_heatmap(ldmks, 224, 224)
        # The heatmap is generated according to the landmarks (224, 224, 5)
        ldmks_heatmap= htmpinone(ldmks_heatmap, 224, 224)  # Turn 5 heatmap into one (224, 224, 3)
        ldmks_heatmap = ldmks_heatmap.transpose((2, 0, 1))  # transpose to (3, 224, 224)
        totensor = torchvision.transforms.ToTensor()
        img = totensor(img)
        ldmks_heatmap = torch.tensor(ldmks_heatmap)

        return img, ldmks_heatmap

def htmp2array(heatmap, width, height):
    """
    Convert the heatmap images to np.array
    :param heatmap: the heatmap image
    :param width: the width of the image
    :param height: the height of the image
    :return: array of the heatmap
    """
    # heatmap_ = heatmap * 256
    heatmap_ = np.array(heatmap)
    heatmap_3way = np.zeros((width, height, 3))
    heatmap_3way[:, :, 0] = heatmap_
    heatmap_3way[:, :, 1] = heatmap_
    heatmap_3way[:, :, 2] = heatmap_

    return heatmap_3way

def htmpinone(htmps, width, height):
    """
    The heatmaps are respectively discribed in 5 images,
    now combine the 5 images into 1
    :param htmps: 5 heatmap images array: (5, 224, 224)
    :param width: the width of the image
    :param height: the height of the image
    :return: One image with 5 heatmaps: array (3, 224, 224)
    """
    htmp1_3way = htmp2array(htmps[:, :, 0], width, height)
    htmp2_3way = htmp2array(htmps[:, :, 1], width, height)
    htmp3_3way = htmp2array(htmps[:, :, 2], width, height)
    htmp4_3way = htmp2array(htmps[:, :, 3], width, height)
    htmp5_3way = htmp2array(htmps[:, :, 4], width, height)

    allhtmp = htmp1_3way + htmp2_3way + htmp3_3way + htmp4_3way + htmp5_3way

    return allhtmp


def main():
    """
    Testing of the codes
    :return:
    """
    zjs = xmldataset(root='data_center2.txt')
    datalo = DataLoader(zjs)
    # print(len(datalo.dataset))
    for i, (img, ldmks) in enumerate(datalo):
        print(img.shape, ldmks.shape)


if __name__ == '__main__':
    main()
