import matplotlib.pyplot as plt
import numpy as np
import cv2

def CenterLabelHeatMap(img_width, img_height, c_x, c_y, sigma):
    X1 = np.linspace(1, img_width, img_width)
    Y1 = np.linspace(1, img_height, img_height)
    [X, Y] = np.meshgrid(X1, Y1)
    X = X - c_x
    Y = Y - c_y
    D2 = X * X + Y * Y
    E2 = 2.0 * sigma * sigma
    Exponent = D2 / E2
    heatmap = np.exp(-Exponent)
    return heatmap

if __name__ == '__main__':
    d=[]
    a1=CenterLabelHeatMap(100,100,50,50,13)
    a2=CenterLabelHeatMap(100,100,10,10,13)
    d.append(a1)
    d.append(a2)
    d=np.stack(d)
    print(d.shape)
    cv2.imshow('d',d[0])
    cv2.waitKey(0)