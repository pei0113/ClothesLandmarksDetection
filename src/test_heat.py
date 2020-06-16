# -*- coding: UTF-8 -*-
import os
import cv2
import numpy as np

from PIL import Image
from df_dataset_visible import DFDatasets
from networks import DenseNet121Heat, GHCU_visible, LVNet
from evaluate import calculate_NMSE, evaluate_visible
from scipy.ndimage import gaussian_filter

import torch

DEBUG_MODE = False
VISUALIZE_MODE = False
EVALUATE_MODE = True
use_gpu = True
root = '../'
ckpt_path = root + 'checkpoints/v8/epoch_80.pth'
img_path = root + 'data/train/'
test_txt = root + 'data/upper/test_list.txt'
bbox_txt = root + 'data/Anno/list_bbox.txt'

# load data list
test_dataset = DFDatasets(test_txt, bbox_txt, DEBUG_MODE, root)
test_loader = torch.utils.data.DataLoader(batch_size=1, dataset=test_dataset, num_workers=1)

# load model
model = DenseNet121Heat()

if use_gpu:
    model.cuda()

# load weight
model.load_state_dict(torch.load(ckpt_path))

error_total = 0
acc_total = 0
# predict
for i, inputs in enumerate(test_loader):
    im = inputs['im_name'][0]
    im = Image.open(os.path.join('../data', im))
    im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = inputs['bbox_tl']
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = int(bbox_x1), int(bbox_y1), int(bbox_x2), int(bbox_y2)
    bbox_h, bbox_w = bbox_y2 - bbox_y1, bbox_x2 - bbox_x1
    # x_gt, y_gt, conf_nocut_gts, conf_vis_gts = inputs['label_gt']
    x_gts, y_gts, conf_vis_gts = inputs['label_gt']

    im_tensor = inputs['im_tensor'].cuda()
    output_heat = model(im_tensor)
    output_heat = output_heat.data.cpu().numpy()[0]

    out_xs = []
    out_ys = []
    for j in range(0, 6):
        canvas = im.copy()
        out_heat = output_heat[j]
        #out_heat = gaussian_filter(out_heat, 1)

        # predict x, y, vis from heatmap
        out_heat_abs = np.abs(out_heat)
        out_y, out_x = np.where(out_heat_abs == np.amax(out_heat_abs))
        out_y, out_x = out_y[0], out_x[0]

        out_xs.append(out_x/224)
        out_ys.append(out_y/224)

        canvas_y, canvas_x = int(out_y * bbox_h / 224 + bbox_y1), int(out_x * bbox_w / 224+ bbox_x1)
        cv2.circle(canvas, (canvas_x, canvas_y), 3, (0, 0, 255), -1)
        cv2.putText(canvas, str(j), (canvas_x, canvas_y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2, 2)

        if VISUALIZE_MODE:
            cv2.imshow('img', canvas)
            cv2.waitKey(0)

    if EVALUATE_MODE:
        out_xs = np.array(out_xs)
        out_ys = np.array(out_ys)
        # evaluate
        landmark_gt = inputs['label_gt']
        landmark_gt = [n[0].cpu().numpy() for n in landmark_gt]
        out_numpy = [[], out_xs, out_ys]
        error = calculate_NMSE(gts=landmark_gt, pds=out_numpy)
        error_total += error
        print(' [*] Evaluate: {} / {}'.format(i, len(test_loader)))

if EVALUATE_MODE:
    error_avg = error_total / len(test_loader)
    print(' [*] Average NMSE = {}'.format(error_avg))
