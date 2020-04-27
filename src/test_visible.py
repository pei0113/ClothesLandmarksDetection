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
VISUALIZE_MODE = True
EVALUATE_MODE = False
use_gpu = True
root = '../'
ckpt_path = root + 'checkpoints/epoch_80.pth'
img_path = root + 'data/train/'
test_txt = root + 'data/upper/test_list.txt'
bbox_txt = root + 'data/Anno/list_bbox.txt'

# load data list
test_dataset = DFDatasets(test_txt, bbox_txt, DEBUG_MODE)
test_loader = torch.utils.data.DataLoader(batch_size=1, dataset=test_dataset, num_workers=1)

# load model
model = LVNet()

if use_gpu:
    model.cuda()

# load weight
model.load_state_dict(torch.load(ckpt_path))

error_total = 0
acc_total = 0
# predict
for i, inputs in enumerate(test_loader):
    im = inputs['im_name'][0]
    im = Image.open(os.path.join('data', im))
    im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = inputs['bbox_tl']
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = int(bbox_x1), int(bbox_y1), int(bbox_x2), int(bbox_y2)
    bbox_h, bbox_w = bbox_y2 - bbox_y1, bbox_x2 - bbox_x1
    # x_gt, y_gt, conf_nocut_gts, conf_vis_gts = inputs['label_gt']
    x_gts, y_gts, conf_vis_gts = inputs['label_gt']

    im_tensor = inputs['im_tensor'].cuda()
    output = model(im_tensor)
    # out_coord, out_conf_nocut, out_conf_vis = output
    out_coord, out_conf_vis = output
    out_x, out_y = torch.split(out_coord, 6, dim=1)

    if EVALUATE_MODE:
        # evaluate
        landmark_gt = inputs['label_gt']        # [x, y, vis]
        landmark_gt = [n.cpu().numpy() for n in landmark_gt]
        out_numpy = [n[0].cpu().detach().numpy() for n in [out_conf_vis, out_x, out_y]]
        error = calculate_NMSE(gts=landmark_gt[:2], pds=out_numpy[1:])
        error_total += error
        acc_vis = evaluate_visible(landmark_gt[2], out_numpy[0])
        acc_total += acc_vis
        print(' [*] Evaluate: {} / {}'.format(i, len(test_loader)))

    if VISUALIZE_MODE:
        for j in range(0, 6):
            canvas = im.copy()
            # conf_nocut, conf_vis = out_conf_nocut[0][j], out_conf_vis[0][j]
            conf_vis = out_conf_vis[0][j]
            # conf_nocut_gt, conf_vis_gt = conf_nocut_gts[0][j], conf_vis_gts[0][j]
            conf_vis_gt = conf_vis_gts[0][j]

            # if conf_nocut_gt == 0:

            x, y = out_x[0][j], out_y[0][j]
            x, y = (x * bbox_w + bbox_x1), (y * bbox_h + bbox_y1)
            x_gt, y_gt = x_gts[0][j], y_gts[0][j]
            x_gt, y_gt = (x_gt * bbox_w + bbox_x1), (y_gt * bbox_h + bbox_y1)

            canvas = cv2.circle(canvas, (x, y), 3, (0, 0, 255), -1)
            canvas = cv2.circle(canvas, (x_gt, y_gt), 3, (0, 255, 0), -1)
            # canvas = cv2.putText(canvas, 'conf_nocut:{:.2f}, conf_vis:{:.2f}'.format(conf_nocut.item(), conf_vis), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2, 2)
            canvas = cv2.putText(canvas, 'conf_vis:{:.2f}'.format(conf_vis), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2, 2)
            canvas = cv2.putText(canvas, 'gt:{:.2f}'.format(conf_vis_gt), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2, 2)
            cv2.imshow('img', canvas)
            cv2.waitKey(0)
            cv2.imwrite('result/v17/im{}_{}.jpg'.format(i, j), canvas)

if EVALUATE_MODE:
    error_avg = error_total / len(test_loader)
    print(' [*] Average NMSE = {}'.format(error_avg))
    acc_avg = acc_total / (len(test_loader) * 6)
    print(' [*] Visible Accuracy = {}'.format(acc_avg))
