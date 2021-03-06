# -*- coding: UTF-8 -*-
import os
import cv2
import numpy as np

from PIL import Image
from df_dataset_bbox import DFDatasets
from networks import HRNetFashionNet, DenseNet121Heat
from scipy.ndimage import gaussian_filter

import torch

DEBUG_MODE = True
use_gpu = True
checkpoints_path = 'checkpoints/v7/epoch_100.pth'
img_path = 'data/test/'
test_txt = 'data/upper/test_list.txt'
bbox_txt = 'data/Anno/list_bbox.txt'

# load data list
test_dataset = DFDatasets(test_txt, bbox_txt, DEBUG_MODE)
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
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = inputs['bbox_tl']
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = int(bbox_x1), int(bbox_y1), int(bbox_x2), int(bbox_y2)
    bbox_h, bbox_w = bbox_y2 - bbox_y1, bbox_x2 - bbox_x1

    im_tensor = inputs['im_tensor'].cuda()
    [vis_gt, x_gt, y_gt] = inputs['label_gt']
    [vis_gt, x_gt, y_gt] = vis_gt.cuda(), x_gt.cuda(), y_gt.cuda()
    vis_gt, x_gt, y_gt = vis_gt[0], x_gt[0], y_gt[0]

    output_heat = model(im_tensor)[0]
    output_heat = output_heat.data.cpu().numpy()
    output_heat = gaussian_filter(output_heat, 1)

    for j in range(0, 6):
        out_heat = output_heat[j]
        cv2.imshow('heatmap', out_heat*255)
        cv2.waitKey(0)
        out_y, out_x = np.where(out_heat == np.amax(out_heat))
        try:
            out_y, out_x = int(out_y*bbox_h/224 + bbox_y1), int(out_x*bbox_w/244 + bbox_x1)
            cv2.circle(im, (out_x, out_y), 3, (0, 0, 255), -1)
            cv2.putText(im, str(j), (out_x, out_y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2, 2)
        except:
            continue

        # if vis_gt[j] == 1:
        #     cv2.circle(im, (int(x_gt[j]), int(y_gt[j])), 3, (0, 255, 0), -1)
        cv2.imshow('img', im)
        cv2.waitKey(0)
        # cv2.imwrite()
