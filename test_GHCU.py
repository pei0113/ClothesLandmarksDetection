# -*- coding: UTF-8 -*-
import os
import cv2
import numpy as np

from PIL import Image
from df_dataset_bbox import DFDatasets
from networks import GHCU, DenseNet121Heat
from scipy.ndimage import gaussian_filter

import torch

DEBUG_MODE = False
use_gpu = True
ckpt_path_HEAT = 'checkpoints/v8/epoch_100.pth'
ckpt_path_GHCU = 'checkpoints/epoch_70.pth'
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
    im = Image.open(os.path.join('data', im_name))
    im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = inputs['bbox_tl']
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = int(bbox_x1), int(bbox_y1), int(bbox_x2), int(bbox_y2)
    bbox_h, bbox_w = bbox_y2 - bbox_y1, bbox_x2 - bbox_x1

    im_tensor = inputs['im_tensor'].cuda()
    output_heat = model_HEAT(im_tensor)
    # input2 = torch.cat((output_heat, im_tensor), 1)
    output = model_GHCU(output_heat)
    out_lm = torch.split(output, 2, 1)
    # out_viss = output[1]

    for j in range(0, 6):
        out_x, out_y = out_lm[j][0][0], out_lm[j][0][1]
        try:
            out_y, out_x = int(out_y*bbox_h + bbox_y1), int(out_x*bbox_w + bbox_x1)
            cv2.circle(im, (out_x, out_y), 3, (0, 0, 255), -1)
            cv2.putText(im, str(j), (out_x, out_y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2, 2)
        except:
            continue

        # if vis_gt[j] == 1:
        #     cv2.circle(im, (int(x_gt[j]), int(y_gt[j])), 3, (0, 255, 0), -1)
    cv2.imshow('img', im)
    cv2.waitKey(0)
    # cv2.imwrite('result/v9/'+im_name.split('/')[1], im)
