import os
import cv2
import numpy as np

from PIL import Image
from df_dataset import DFDatasets
from networks import Vgg16FashionNet, DenseNet121FashionNet
from torchvision.models import vgg19, densenet121

import torch


output_classes = 18
use_gpu = True
checkpoints_path = 'checkpoints/epoch_50.pth'
img_path = 'data/test/'
test_txt = 'data/upper/test_list.txt'

# load data list
test_dataset = DFDatasets(test_txt)
test_loader = torch.utils.data.DataLoader(batch_size=1, dataset=test_dataset, num_workers=1)

# load denseNet model
# model = densenet121(pretrained=True)
# num_features = model.classifier.in_features
# features = nn.Linear(num_features, output_classes)
# model.classifier = features

# load vgg16 model
# model = vgg19()
# num_features = model.classifier[6].in_features
# features = list(model.classifier.children())[:-1] # Remove last layer
# features.extend([nn.Linear(num_features, output_classes)]) # Add our layer with 4 outputs
# model.classifier = nn.Sequential(*features) # Replace the model classifier

# load vgg16 model
model = DenseNet121FashionNet()

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
    [vis, x, y] = inputs['labels']
    [vis, x, y] = vis.cuda(), x.cuda(), y.cuda()
    vis, x, y = vis[0], x[0], y[0]

    output = model(im_tensor)
    out_landmarks = output[0]
    out_vis = output[1]
    out_x, out_y = torch.split(out_landmarks, 6, 1)
    out_x, out_y = out_x[0], out_y[0]

    # output = F.sigmoid(output)
    # out_vis, out_x, out_y = torch.split(output, 6, 1)
    # out_vis, out_x, out_y = out_vis[0], out_x[0], out_y[0]

    for j in range(0, 6):
        if out_vis[0][j] >= 0.5:
            cv2.circle(im, (int(out_x[j]*w), int(out_y[j]*h)), 3, (0, 0, 255), -1)
        if vis[j] == 1:
            cv2.circle(im, (int(x[j]*w/224), int(y[j]*h/224)), 3, (0, 255, 0), -1)
        cv2.imshow('img', im)
        cv2.waitKey(0)
