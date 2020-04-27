# -*- coding: UTF-8 -*-
import os
import cv2
import numpy as np

from PIL import Image
from df_dataset_GHCU import DFDatasets
from networks import GHCU, DenseNet121Heat
from evaluate import calculate_NMSE
from scipy.ndimage import gaussian_filter

import torch

DEBUG_MODE = False
VISUALIZE_MODE = True
EVALUATE_MODE = False
use_gpu = True
root = '../'
ckpt_path_HEAT = root + 'checkpoints/v8/epoch_100.pth'
ckpt_path_GHCU = root + 'checkpoints/v10/epoch_70.pth'
img_path = root + 'data/test/'
test_txt = root + 'data/upper/test_list.txt'
bbox_txt = root + 'data/Anno/list_bbox.txt'

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

error_total = 0
# predict
for i, inputs in enumerate(test_loader):
    im_name = inputs['im_name'][0]
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
    # input2 = torch.cat((output_heat, im_tensor), 1)
    out_lm, out_viss = model_GHCU(output_heat)
    out_xs, out_ys = torch.split(out_lm, 6, 1)
    out_xs, out_ys = out_xs[0], out_ys[0]
    out_viss = out_viss[0]

    if EVALUATE_MODE:
        # evaluate
        landmark_gt = inputs['landmark_gt']
        landmark_gt = [n[0].cpu().numpy() for n in landmark_gt]
        out_numpy = [n.cpu().detach().numpy() for n in [out_viss, out_xs, out_ys]]
        error = calculate_NMSE(gts=landmark_gt, pds=out_numpy)
        error_total += error
        print(' [*] Evaluate: {} / {}'.format(i, len(test_loader)))

    if VISUALIZE_MODE:
        canvas = im.copy()
        for j in range(0, 6):
            out_x, out_y = out_xs[j], out_ys[j]
            out_vis = float(out_viss[j])
            try:
                out_y, out_x = int(out_y*bbox_h + bbox_y1), int(out_x*bbox_w + bbox_x1)
                cv2.circle(canvas, (out_x, out_y), 3, (0, 0, 255), -1)
                cv2.putText(canvas, str(j), (out_x, out_y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2, 2)
                cv2.putText(canvas, 'vis"{}'.format(out_vis), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2, 2)
            except:
                continue

        cv2.imshow('img', canvas)
        cv2.waitKey(0)
        # cv2.imwrite('result/v10/'+im_name.split('/')[1], canvas)

if EVALUATE_MODE:
    error_avg = error_total / len(test_loader)
    print(' [*] Average NMSE = {}'.format(error_avg))
