import numpy as np
import math
import cv2.cv2 as cv2


def get_heatmap(annos, height, width):
    """
    Parameters
    - annos： list of the labels [
                            x1, y1,
                            x2, y2,
                            x3, y3,
                            x4, y4,
                            x5, y5,
                                ]
    - height：The height of the image
    - width: The width of the image
    Returns
    - heatmap
    """

    coors = []
    coors.append([annos[0], annos[1]])
    coors.append([annos[2], annos[3]])
    coors.append([annos[4], annos[5]])
    coors.append([annos[6], annos[7]])
    coors.append([annos[8], annos[9]])

    num_joints = 5

    joints_heatmap = np.zeros((num_joints, height, width), dtype=np.float32)

    for i, points in enumerate(coors):
        if points[0] < 0 or points[1] < 0:
            continue
        joints_heatmap = put_heatmap(joints_heatmap, i, points, 4)

    joints_heatmap = joints_heatmap.transpose((1, 2, 0))

    mapholder = []
    for i in range(0, 5):
        a = cv2.resize(np.array(joints_heatmap[:, :, i]), (height, width))
        mapholder.append(a)
    mapholder = np.array(mapholder)
    joints_heatmap = mapholder.transpose((1, 2, 0))

    return joints_heatmap.astype(np.float16)


def put_heatmap(heatmap, plane_idx, center, sigma):
    """
    Parameters
    -heatmap:
    - plane_idx：The sequence of the landmarks
    - center： the positions of the landmarks
    - sigma: the parameter to generate Gauss Heatmap
    Returns
    - heatmap
    """

    center_x, center_y = center
    _, height, width = heatmap.shape[:3]

    th = 4.6052
    delta = math.sqrt(th * 2)

    x0 = int(max(0, center_x - delta * sigma + 0.5))
    y0 = int(max(0, center_y - delta * sigma + 0.5))

    x1 = int(min(width - 1, center_x + delta * sigma + 0.5))
    y1 = int(min(height - 1, center_y + delta * sigma + 0.5))

    exp_factor = 1 / 2.0 / sigma / sigma

    ## fast - vectorize
    arr_heatmap = heatmap[plane_idx, y0:y1 + 1, x0:x1 + 1]
    y_vec = (np.arange(y0, y1 + 1) - center_y) ** 2
    x_vec = (np.arange(x0, x1 + 1) - center_x) ** 2
    xv, yv = np.meshgrid(x_vec, y_vec)
    arr_sum = exp_factor * (xv + yv)
    arr_exp = np.exp(-arr_sum)
    arr_exp[arr_sum > th] = 0
    heatmap[plane_idx, y0:y1 + 1, x0:x1 + 1] = np.maximum(arr_heatmap, arr_exp)
    return heatmap


def main():
    """
    Testing of the codes
    """
    # image size is 224 * 224
    # number of the landmarks is 5
    # heatmap = np.zeros((10, 224, 224))
    # print(heatmap)
    # a = np.arange(2)
    # b = put_heatmap(heatmap, 1, a, 1)
    points = [100.0, 20.0, 30.0, 40.0, 50.0, 200.0, 70.0, 80.0, 150.0, 10.0]
    c = get_heatmap(points, 224, 224)
    # c = c.transpose((2, 0, 1))
    c5 = c[:, :, 4]
    print(c5)
    # img1 = Image.fromarray(np.uint8(c3*255))
    # img1.show()
    # print(c1.shape)


if __name__ == '__main__':
    main()
