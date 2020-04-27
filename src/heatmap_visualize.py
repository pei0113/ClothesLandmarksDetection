# -*- coding: UTF-8 -*-
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from PIL import Image
from df_dataset_bbox import DFDatasets
from networks import GHCU, DenseNet121Heat
from scipy.ndimage import gaussian_filter

import torch

DEBUG_MODE = True
use_gpu = True
ckpt_path_HEAT = 'checkpoints/v8/epoch_100.pth'
ckpt_path_GHCU = 'checkpoints/v10/epoch_70.pth'
img_path = 'data/test/'
test_txt = 'data/upper/test_list.txt'
bbox_txt = 'data/Anno/list_bbox.txt'

# load data list
test_dataset = DFDatasets(test_txt, bbox_txt, DEBUG_MODE)
test_loader = torch.utils.data.DataLoader(batch_size=1, dataset=test_dataset, num_workers=1)

# load model
model_HEAT = DenseNet121Heat()
model_GHCU = GHCU()
if use_gpu:
    model_HEAT.cuda()
    model_GHCU.cuda()

# load weight
model_HEAT.load_state_dict(torch.load(ckpt_path_HEAT))
model_GHCU.load_state_dict(torch.load(ckpt_path_GHCU))

# predict
for i, inputs in enumerate(test_loader):
    im_name = inputs['im_name'][0]
    # im_name = 'test/img_00103035.jpg'
    im = Image.open(os.path.join('data', im_name))
    im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = inputs['bbox_tl']
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = int(bbox_x1), int(bbox_y1), int(bbox_x2), int(bbox_y2)
    bbox_h, bbox_w = bbox_y2 - bbox_y1, bbox_x2 - bbox_x1

    # [OUTPUT X & Y]
    # im_tensor = inputs['im_tensor'].cuda()
    # output_heat = model_HEAT(im_tensor)
    # output = model_GHCU(output_heat)
    # out_lm = torch.split(output, 2, 1)
    #
    # for j in range(0, 6):
    #     out_x, out_y = out_lm[j][0][0], out_lm[j][0][1]
    #     try:
    #         out_y, out_x = int(out_y*bbox_h + bbox_y1), int(out_x*bbox_w + bbox_x1)
    #         cv2.circle(im, (out_x, out_y), 3, (0, 0, 255), -1)
    #         cv2.putText(im, str(j), (out_x, out_y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2, 2)
    #     except:
    #         continue

    # [OUTPUT VIS & X & Y]
    im_tensor = inputs['im_tensor'].cuda()
    output_heat = model_HEAT(im_tensor)
    out_lm, out_viss = model_GHCU(output_heat)
    out_xs, out_ys = torch.split(out_lm, 6, 1)
    out_xs, out_ys = out_xs[0], out_ys[0]
    out_viss = out_viss[0]

    canvas = im.copy()
    output_heat = output_heat[0]
    for j in range(0, 6):

        # plot 3D heatmap
        out_heat = output_heat[j].cpu().detach().numpy()
        xy_axis = np.arange(0, 224, 1)
        X, Y = np.meshgrid(xy_axis, xy_axis)
        fig = plt.figure()
        # plt.subplot(321+j)

        ax = Axes3D(fig)
        ax.plot_surface(X, Y, out_heat, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))

        out_x, out_y = out_xs[j], out_ys[j]
        out_vis = float(out_viss[j])
        # cv2.putText(canvas, 'vis:{}'.format(str(out_vis)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2, 2)
        try:
            out_y, out_x = int(out_y*bbox_h + bbox_y1), int(out_x*bbox_w + bbox_x1)
            cv2.circle(canvas, (out_x, out_y), 3, (0, 0, 255), -1)
            cv2.putText(canvas, str(j), (out_x, out_y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2, 2)
        except:
            continue
    plt.show()
    cv2.imshow('img', canvas)
    cv2.waitKey(0)


    # cv2.imwrite('result/v10/'+im_name.split('/')[1], canvas)
