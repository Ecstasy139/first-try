from coordinate2heatmap import *
from transform import *
import heapq

def htmp2array(heatmap, width, height):
    # heatmap_ = heatmap * 256
    heatmap_ = np.array(heatmap)
    heatmap_3way = np.zeros((width, height, 3))
    heatmap_3way[:, :, 0] = heatmap_
    heatmap_3way[:, :, 1] = heatmap_
    heatmap_3way[:, :, 2] = heatmap_

    return heatmap_3way

def htmpinone(htmps, width, height):
    htmp1_3way = htmp2array(htmps[:, :, 0], width, height)
    htmp2_3way = htmp2array(htmps[:, :, 1], width, height)
    htmp3_3way = htmp2array(htmps[:, :, 2], width, height)
    htmp4_3way = htmp2array(htmps[:, :, 3], width, height)
    htmp5_3way = htmp2array(htmps[:, :, 4], width, height)

    allhtmap = htmp1_3way + htmp2_3way + htmp3_3way + htmp4_3way + htmp5_3way

    return allhtmap




#
# img_path = r'part-of-AFLW\already_labeled\image00014.jpg'
# img = Image.open(img_path)
# ldmks = [1064, 1085, 1555, 1061, 1322, 2001, 1388, 2295, 1466, 2738]
# res = Rescale((256, 256))
# rand = RandomCrop((224, 224))
# randro = RandomRotate((-10, 10))
# img, ldmks = res(img, ldmks)
# img, ldmks = rand(img, ldmks)
# # img, ldmks = randro(img, ldmks)
# ldmks_heatmap = get_heatmap(ldmks, 224, 224)
# ldmks_heatmap_ = ldmks_heatmap.transpose((2, 0, 1))
#
# #
# # heatmap_3way1 = htmp2array(ldmks_heatmap[:, :, 0], 224, 224)
# # heatmap_3way2 = htmp2array(ldmks_heatmap[:, :, 1], 224, 224)
# # heatmap_3way3 = htmp2array(ldmks_heatmap[:, :, 2], 224, 224)
# # heatmap_3way4 = htmp2array(ldmks_heatmap[:, :, 3], 224, 224)
# # heatmap_3way5 = htmp2array(ldmks_heatmap[:, :, 4], 224, 224)
#
#
# # heatmap_ = heatmap_3way1 + heatmap_3way2 + heatmap_3way3 + heatmap_3way4 + heatmap_3way5 + img
#
# heatmap_ = htmpinone(ldmks_heatmap, 224, 224)
# # heatmap_ = heatmap_.transpose((2, 0, 1))
# img_heatmap = Image.fromarray(np.uint8(heatmap_))
# img_heatmap.show()





img_path = r'part-of-AFLW\already_labeled\image00014.jpg'

img = Image.open(img_path, 'r')
print(img.size)
ldmks = [1064, 1085, 1555, 1061, 1322, 2001, 1388, 2295, 1466, 2738]

res = Rescale((224, 224))

img1, ldmks = res(img, ldmks)

ldmks_htmp = get_heatmap(ldmks, 224, 224)
ldmks_htmp = htmpinone(ldmks_htmp, 224, 224)
#ldmks = [91, 62, 134, 61, 114, 115, 119, 132, 126, 157]
ldmks_htmp = ldmks_htmp.transpose((2, 0, 1))
htmp1way = ldmks_htmp[1]

htmp1way = htmp1way.ravel()
htmp1way = htmp1way.tolist()
max_list = list(map(htmp1way.index, heapq.nlargest(5, htmp1way)))

points = []
for i in range(5):
    x = max_list[i] % 224
    y = max_list[i] // 224
    points.append(x/224)
    points.append(y/224)
print(points)











