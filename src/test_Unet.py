# -*- coding: UTF-8 -*-
import torch
import os
import sys
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

from df_dataset_bbox import DFDatasets
from networks import DenseNet121Heat, LVUNet5
from evaluate import calculate_NMSE, evaluate_visible
from cu_net import create_cu_net


DEBUG_MODE = False
VISUALIZE_MODE = False
EVALUATE_MODE = True
use_gpu = True
root = '../'
checkpoints_path = root + 'checkpoints/v25/epoch_100.pth'
img_path = root + 'data/test/'
test_txt = root + 'data/upper/test_list.txt'
bbox_txt = root + 'data/Anno/list_bbox.txt'

# load data list
test_dataset = DFDatasets(test_txt, bbox_txt, DEBUG_MODE, root)
test_loader = torch.utils.data.DataLoader(batch_size=1, dataset=test_dataset, num_workers=1)

# load model
model = LVUNet5()
# model = create_cu_net(neck_size=4, growth_rate=32, init_chan_num=128, class_num=6, layer_num=2, order=1, loss_num=1)

if use_gpu:
    model.cuda()

# load weight
model.load_state_dict(torch.load(checkpoints_path))
model.eval()

error_total = 0
acc_total = 0
# predict
for i, inputs in enumerate(test_loader):
    im = inputs['im_name'][0]
    im = Image.open(os.path.join(root + 'data', im))
    im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = inputs['bbox_tl']
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = int(bbox_x1), int(bbox_y1), int(bbox_x2), int(bbox_y2)
    bbox_h, bbox_w = bbox_y2 - bbox_y1, bbox_x2 - bbox_x1

    im_tensor = inputs['im_tensor'].cuda()
    [vis_gts, x_gts, y_gts] = inputs['label_gt']
    vis_gts, x_gts, y_gts = vis_gts[0], x_gts[0], y_gts[0]
    vis_gts, x_gts, y_gts = vis_gts.data.numpy(), x_gts.data.numpy(), y_gts.data.numpy()

    output_heat = model(im_tensor)
    # output_vis = output_vis[0]
    output_heat = output_heat.data.cpu().numpy()[0]
    # output_vis = output_vis.data.cpu().numpy()[0]

    out_xs = []
    out_ys = []
    out_viss = []

    for j in range(0, 6):
        canvas = im.copy()
        out_heat = output_heat[j]
        # out_heat = gaussian_filter(out_heat, 1)
        gt_vis, gt_x, gt_y = vis_gts[j], x_gts[j], y_gts[j]

        # plt.clf()
        # plt.imshow(out_heat)
        # plt.colorbar()
        # plt.show()

        # EVALUATE METHOD: heatmap sum
        # heat_sum = np.sum(out_heat)
        # out_vis = 1 if heat_sum > 0 else 0
        # if out_vis == 1:
        #     out_y, out_x = np.where(out_heat == np.amax(out_heat))
        # else:
        #     out_y, out_x = np.where(out_heat == np.amin(out_heat))

        # EVALUATE METHOD: heatmap amplitude max
        out_heat_abs = np.abs(out_heat)
        out_y, out_x = np.where(out_heat_abs == np.amax(out_heat_abs))
        out_vis = 1 if out_heat[out_y, out_x] > 0 else 0

        out_y, out_x = out_y[0], out_x[0]
        canvas_y, canvas_x = int(out_y*bbox_h/224 + bbox_y1), int(out_x*bbox_w/224 + bbox_x1)
        out_xs.append(out_x/224)
        out_ys.append(out_y/224)
        out_viss.append(out_vis)

        cv2.circle(canvas, (canvas_x, canvas_y), 3, (0, 0, 255), -1)
        cv2.putText(canvas, str(float(out_vis)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2, 2)
        cv2.circle(canvas, (gt_x, gt_y), 3, (255, 0, 0), -1)
        cv2.putText(canvas, str(float(gt_vis)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 2, 2)
        cv2.putText(canvas, str(j), (canvas_x, canvas_y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2, 2)

        # cv2.imwrite('../result/v25/{}_{}.jpg'.format(i, j), canvas)

        if VISUALIZE_MODE:
            cv2.imshow('img', canvas)
            cv2.waitKey(0)

    if EVALUATE_MODE:
        out_xs = np.array(out_xs)
        out_ys = np.array(out_ys)
        out_viss = np.array(out_viss)
        # evaluate
        landmark_gt = inputs['landmark_gt']
        landmark_gt = [n[0].cpu().numpy() for n in landmark_gt]
        out_numpy = [out_viss, out_xs, out_ys]
        error = calculate_NMSE(gts=landmark_gt, pds=out_numpy)
        error_total += error
        acc_vis = evaluate_visible(landmark_gt[0], out_numpy[0])
        acc_total += acc_vis
        print(' [*] Evaluate: {} / {}'.format(i, len(test_loader)))

if EVALUATE_MODE:
    error_avg = error_total / len(test_loader)
    print(' [*] Average NMSE = {}'.format(error_avg))
    acc_avg = acc_total / (len(test_loader) * 6)
    print(' [*] Visible Accuracy = {}'.format(acc_avg))
