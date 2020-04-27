import os
import sys
import numpy as np
from time import time

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(".."))

from networks import DenseNet121Heat, GHCU
from df_dataset_GHCU import DFDatasets


def criterion_GHCU(vis, x, y, output):
    out_x, out_y = torch.split(output[0], 6, 1)
    out_vis = output[1]

    # vis_criterion = nn.MSELoss()
    # vis_loss = torch.abs(out_vis - vis)             # [32, 6]

    loss_x = torch.abs(out_x - x)                   # [32, 6]
    loss_y = torch.abs(out_y - y)                   # [32, 6]
    loss_lm = loss_x + loss_y                       # [32, 6]
    loss_lm = torch.mul(loss_lm, loss_lm)           # [32, 6]
    loss_lm = vis * loss_lm                         # [32, 6]
    loss_lm = torch.sum(loss_lm)/(batch_size*6)     # [1]

    return loss_lm


DEBUG_MODE = False
num_worker = 10
use_gpu = True
lr = 0.001
batch_size = 32
validation_split = 0.2
shuffle_dataset = True
random_seed = 123

root = '../'
checkpoint_path = root + 'checkpoints/v8/epoch_100.pth'
test_txt = root + 'data/upper/train_list.txt'
bbox_txt = root + 'data/Anno/list_bbox.txt'

# load model stage 1(HeatMap generator)
model_HEAT = DenseNet121Heat()
# load model stage 2
model_GHCU = GHCU()

# load data list
train_dataset = DFDatasets(test_txt, bbox_txt, DEBUG_MODE, root)
train_loader = torch.utils.data.DataLoader(batch_size=batch_size, dataset=train_dataset, num_workers=1)

# Creating data indices for training and validation splits:
dataset_size = len(train_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
validation_sampler = SubsetRandomSampler(val_indices)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_worker, sampler=train_sampler)
validation_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_worker, sampler=validation_sampler)

if use_gpu:
    model_HEAT.cuda()
    model_GHCU.cuda()

# load weight
model_HEAT.load_state_dict(torch.load(checkpoint_path))

criterion = nn.MSELoss()
optimizer = optim.Adam(model_GHCU.parameters(), lr=lr)
# scheduler_2 = LambdaLR(optimizer, lr_lambda=[lambda epoch: 0.95 ** epoch])
scheduler = ReduceLROnPlateau(optimizer, mode='min', verbose=True)

total_train_loss = 0
total_val_loss = 0
for epoch in range(200):

    tStart = time()

    for batch_idx, inputs in enumerate(train_loader):
        im_tensor = inputs['im_tensor'].cuda()
        output_heat = model_HEAT(im_tensor)

        # input2 = torch.cat((output_heat, im_tensor), 1)
        output = model_GHCU(output_heat)

        # [ONLY CALCULATE LOSS OF X AND Y COORDINATE]
        # xy_gt = inputs['xy_gt']
        # xy_gt = xy_gt.cuda()
        # loss = criterion(xy_gt, output)

        # [CALCULATE LOSS OF VIS & X & Y]
        [vis_gt, x_gt, y_gt] = inputs['landmark_gt']
        vis_gt, x_gt, y_gt = vis_gt.cuda(), x_gt.cuda(), y_gt.cuda()
        loss = criterion_GHCU(vis_gt, x_gt, y_gt, output)
        
        loss = loss.requires_grad_()
        total_train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        for batch_idx, inputs in enumerate(validation_loader):
            im_tensor = inputs['im_tensor'].cuda()
            output_heat = model_HEAT(im_tensor)

            # input2 = torch.cat((output_heat, im_tensor), 1)
            output = model_GHCU(output_heat)

            # xy_gt = inputs['xy_gt']
            # xy_gt = xy_gt.cuda()
            # loss = criterion(xy_gt, output)

            [vis_gt, x_gt, y_gt] = inputs['landmark_gt_tensor']
            vis_gt, x_gt, y_gt = vis_gt.cuda(), x_gt.cuda(), y_gt.cuda()
            loss = criterion_GHCU(vis_gt, x_gt, y_gt, output)

            loss = loss.requires_grad_()
            total_val_loss += loss

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(validation_loader)
    scheduler.step(avg_val_loss)

    print('==>>> **train** time: {:.3f}, epoch{}, train loss: {:.6f}, validation loss: {:.6f}'.format(time() - tStart, epoch + 1, avg_train_loss, avg_val_loss))

    total_train_loss = 0
    total_val_loss = 0

    if (epoch + 1) % 10 == 0:
        torch.save(model_GHCU.state_dict(), 'checkpoints/epoch_' + str(epoch + 1) + '.pth')



