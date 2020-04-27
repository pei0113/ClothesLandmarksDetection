import os
import cv2
import numpy as np

from PIL import Image
from df_dataset_heat import DFDatasets
from networks import HRNetFashionNet, DenseNet121Heat
from scipy.ndimage import gaussian_filter

import torch


use_gpu = True
root = '../'
checkpoints_path = root + 'checkpoints/v6/epoch_100.pth'
img_path = root + 'data/test/'
test_txt = root + 'data/upper/test_list.txt'

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
    output_heat = model(im_tensor)
    output_heat = output_heat[0]
    output_heat = output_heat.data.cpu().numpy()
    # output_heat = gaussian_filter(output_heat, 1)

    for j in range(0, 6):
        out_heat = output_heat[j]
        cv2.imshow('heatmap', out_heat*255)
        cv2.waitKey(0)
        out_y, out_x = np.where(out_heat == np.amax(out_heat))
        try:
            out_x, out_y = int(out_x[0]*w/224), int(out_y[0]*h/224)
            cv2.circle(im, (out_x, out_y), 3, (0, 0, 255), -1)
            cv2.putText(im, str(j), (out_x, out_y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2, 2)
        except:
            continue

        cv2.imshow('img', im)
        cv2.waitKey(0)
