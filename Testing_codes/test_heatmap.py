import os

import torch
from PIL import ImageDraw
from model_heatmap import *
from dataloader_heatmap import *
from torchvision.utils import save_image

net = ResNet()


weights = 'params/net_heatmap.pth'
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('successfully')
else:
    print('no loading')
net.eval()
for j in os.listdir('../test_image'):
    img = Image.open(os.path.join('../test_image', j)).resize((224, 224))
    draw = ImageDraw.Draw(img)
    img_data = transform.tf(img)
    img_data = torch.unsqueeze(img_data, dim=0)
    out = net(img_data)
    out = out.squeeze()
    d = torch.max_pool2d(out, 80).squeeze()
    print(d)
    # h,w=np.where(out[0]==out[0].max())
    # print(out[0][h[0]][w[0]])
    rst = []
    for i in range(5):
        h, w = np.where(out[i] == out[i].max())

        # rst.append((w[0],h[0]))
        # print(np.where(out[1]==out[1].max()))
        # d=torch.where(out[1]==0.7949)
        # print(d)
        # c=torch.max(out[1])
        # print(c)

        draw.ellipse((w[0]-3,h[0]-3,w[0]+3,h[0]+3),(255,0,0))
    img.save(f'test_result/{j}')
