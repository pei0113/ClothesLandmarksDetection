# -*- coding: UTF-8 -*-
import torch
import os
import sys
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

from df_dataset_vton import DFDatasets
from networks import DenseNet121Heat, LVUNet5, LVUNet_GHCU, LVUNet_GHCU2, LVUNet_GHCU3, GHCU
from evaluate import calculate_NMSE, evaluate_visible
from cu_net import create_cu_net

import json
bbox_json = '/media/ford/新增磁碟區/pei/person_landmark_detection/coco-data/my_annot/lm_vis.json'
with open(bbox_json, 'r') as json_file:
    json_data = json.loads(json_file.read())


DEBUG_MODE = False
VISUALIZE_MODE = True
EVALUATE_MODE = False
E_by_heat = False
use_gpu = True

root = '/media/ford/新增磁碟區/pei/deepFashion1-landmarks-detection/'
checkpoints_HEAT = root + 'checkpoints/v25/epoch_100.pth'
checkpoints_GHCU = root + 'checkpoints/v27-3/epoch_100.pth'

data_root = '/media/ford/新增磁碟區/pei/cp-vton-master/data/train/'
image_path = data_root + 'image/'
bbox_path = data_root + 'bbox/'
n_keypoints = 6

# load data list
test_dataset = DFDatasets(image_path, bbox_path, DEBUG_MODE, root)
test_loader = torch.utils.data.DataLoader(batch_size=1, dataset=test_dataset, num_workers=0)

# load model
model_HEAT = LVUNet5()
model_GHCU = GHCU()

if use_gpu:
    model_HEAT.cuda()
    model_GHCU.cuda()

# load weight
model_HEAT.load_state_dict(torch.load(checkpoints_HEAT))
model_GHCU.load_state_dict(torch.load(checkpoints_GHCU))
model_HEAT.eval()
model_GHCU.eval()

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

    output_heat = model_HEAT(im_tensor)
    output_loc, output_vis = model_GHCU(output_heat)
    output_heat = output_heat.data.cpu().numpy()[0]
    output_loc = output_loc.data.cpu().numpy()[0]
    output_vis = output_vis.data.cpu().numpy()[0]

    out_xs = []
    out_ys = []
    out_viss = []
    canvas = im.copy()
    for j in range(0, 6):

        out_heat = output_heat[j]
        out_heat = gaussian_filter(out_heat, 1)

        # plt.clf()
        # plt.imshow(out_heat)
        # plt.colorbar()
        # plt.show()
        # # plt.savefig('../result/v25_heat/{}_{}.jpg'.format(i, j))

        if E_by_heat:
            # predict x, y, vis from heatmap
            out_heat_abs = np.abs(out_heat)
            out_y, out_x = np.where(out_heat_abs == np.amax(out_heat_abs))
            out_y, out_x = out_y[0], out_x[0]
            out_vis = 1 if out_heat[out_y, out_x] > 0 else 0
            out_y, out_x = out_y/224, out_x/224
        else:
            # predict x, y, vis from location regression
            out_x, out_y = output_loc[j * 2:j*2 + 2]
            out_vis = output_vis[j]
            out_vis = 1 if out_vis > 0.5 else 0

        canvas_y, canvas_x = int(out_y*bbox_h + bbox_y1), int(out_x*bbox_w + bbox_x1)
        out_xs.append(out_x)
        out_ys.append(out_y)
        out_viss.append(out_vis)

        cv2.circle(canvas, (canvas_x, canvas_y), 3, (0, 0, 255), -1)
        cv2.putText(canvas, str(int(out_vis)), (canvas_x,canvas_y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2, 2)
        #cv2.putText(canvas, str(j), (canvas_x, canvas_y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2, 2)

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
