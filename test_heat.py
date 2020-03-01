import os
import cv2
import numpy as np

from PIL import Image
from df_dataset_heatmap import DFDatasets
from networks import HRNetFashionNet, DenseNet121Heat

import torch


use_gpu = True
checkpoints_path = 'checkpoints/epoch_100.pth'
img_path = 'data/test/'
test_txt = 'data/upper/test_list.txt'

# load data list
test_dataset = DFDatasets(test_txt)
test_loader = torch.utils.data.DataLoader(batch_size=1, dataset=test_dataset, num_workers=1)

# load model
model = DenseNet121Heat()

if use_gpu:
    model.cuda()

# load weight
model.load_state_dict(torch.load(checkpoints_path))

# predict
for i, inputs in enumerate(test_loader):
    im = inputs['im_name'][0]
    im = Image.open(os.path.join('data', im))
    im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    h, w = im.shape[:2]

    im_tensor = inputs['im_tensor'].cuda()
    [vis_gt, x_gt, y_gt] = inputs['label_gt']
    [vis_gt, x_gt, y_gt] = vis_gt.cuda(), x_gt.cuda(), y_gt.cuda()
    vis_gt, x_gt, y_gt = vis_gt[0], x_gt[0], y_gt[0]

    output_heat, output_vis = model(im_tensor)
    output_heat, output_vis = output_heat[0], output_vis[0]
    output_heat[0].data.cpu().numpy()

    # output = F.sigmoid(output)
    # out_vis, out_x, out_y = torch.split(output, 6, 1)
    # out_vis, out_x, out_y = out_vis[0], out_x[0], out_y[0]

    for j in range(0, 6):
        out_heat = output_heat[j].data.cpu().numpy()
        # cv2.imshow('heat', np.concatenate(out_heat, out_heat), axis=2)
        # cv2.waitKey(0)
        out_vis = output_vis[j]

        out_y, out_x = np.where(out_heat == np.amax(out_heat))
        try:
            cv2.circle(im, (int(out_x[0]*w/224), int(out_y[0]*h/224)), 3, (0, 0, 255), -1)
            cv2.putText(im, 'vis: {}'.format(int(out_vis)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1, 2)
        except:
            continue

        # if vis_gt[j] == 1:
        #     cv2.circle(im, (int(x_gt[j]), int(y_gt[j])), 3, (0, 255, 0), -1)
        cv2.imshow('img', im)
        cv2.waitKey(0)
