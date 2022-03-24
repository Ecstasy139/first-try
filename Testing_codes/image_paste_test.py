from coordinate2heatmap import *
from dataloader_heatmap import *
from transform import *

img1_path = r'../part-of-AFLW/already_labeled/image00039.jpg'
img2_path = r'../part-of-AFLW/already_labeled/image00040.jpg'

img = Image.open(img1_path, 'r')
ldmks1 = [546, 539, 740, 550, 633, 874, 617, 982, 633, 1226]
w, h = img.size
res1 = Rescale((224, 224))
res2 = torchvision.transforms.Resize((h, w))
# randc = RandomCrop((224, 224))
totensor = torchvision.transforms.ToTensor()

img1, ldmks1 = res1(img, ldmks1)
# img1, ldmks1 = randc(img1, ldmks1)

ldmks1_htmp = get_heatmap(ldmks1, 224, 224)
ldmks1_htmp = htmpinone(ldmks1_htmp, 224, 224)
ldmks1_htmp = Image.fromarray(np.uint8(ldmks1_htmp * 255))
ldmks1_htmp = res2(ldmks1_htmp)
ldmks1_htmp = np.array(ldmks1_htmp)
img = np.array(img)

final_image = ldmks1_htmp + img
final_image = Image.fromarray(final_image)
final_image.show()




