# -*- coding: UTF-8 -*-
import os
import sys
import numpy as np
from time import time
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torchsummary import summary

sys.path.append(os.path.abspath(".."))

from df_dataset_bbox import DFDatasets
from networks import LVUNet_GHCU3, GHCU, LVUNet5
from cu_net import create_cu_net
from utils import update_loss, show_epoch_loss, set_random_seed
from loss import FOCAL


def convert_gt(gt_tensor):
    gt_np = gt_tensor.cpu().numpy()
    gt_np = np.where(gt_np == 0, 0.5, 1)
    gt_tensor = torch.FloatTensor(gt_np).cuda()
    return gt_tensor


def criterion(output_loc, output_vis, loc_gt, vis_gt):
    # criterion_heat = nn.MSELoss()
    criterion_loc = nn.MSELoss()
    criterion_vis = nn.BCELoss()
    # loss_heat = criterion_heat(output_heat, heat_gt)
    loss_loc = criterion_loc(output_loc, loc_gt)
    loss_vis = criterion_vis(output_vis, vis_gt)
    loss_total = loss_loc + loss_vis
    loss_dict = {'loc': loss_loc,
                 'vis': loss_vis,
                 'total': loss_total}
    return loss_dict


DEBUG_MODE = False
shuffle_dataset = True
use_gpu = True
num_worker = 3
lr = 0.001
batch_size = 64
validation_split = 0.2
n_ketpoints = 6
set_random_seed(2020)

# tensor board
writer = SummaryWriter()

root = '../'
lm_txt = root + 'data/upper/train_list.txt'
bbox_txt = root + 'data/Anno/list_bbox.txt'
class_names = ["left collar", "right collar", "left sleeve", "right sleeve", "left hem", "right hem"]
checkpoint_path = root + 'checkpoints/v25/epoch_100.pth'

# load data list
train_dataset = DFDatasets(lm_txt, bbox_txt, DEBUG_MODE, root)

# Creating data indices for training and validation splits:
dataset_size = len(train_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

if shuffle_dataset:
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
validation_sampler = SubsetRandomSampler(val_indices)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_worker, sampler=train_sampler)
validation_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_worker, sampler=validation_sampler)

# load model
model_HEAT = LVUNet5()
model_GHCU = GHCU()

if use_gpu:
    model_HEAT.cuda()
    model_GHCU.cuda()

# load weight
model_HEAT.load_state_dict(torch.load(checkpoint_path))
summary(model_HEAT, (3, 224, 224))

# train
optimizer = optim.Adam(model_GHCU.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', verbose=True)

train_loss_dict = {'loc': 0, 'vis': 0, 'total': 0}
valid_loss_dict = {'loc': 0, 'vis': 0, 'total': 0}
loss_arr = ['loc', 'vis', 'total']
for epoch in range(1000):

    tStart = time()
    # scheduler_2.step()

    # train
    for batch_idx, inputs in enumerate(train_loader):
        im = inputs['im_tensor'].cuda()
        [heat_gt, vis_gt] = inputs['labels']
        [heat_gt, vis_gt] = heat_gt.cuda(), vis_gt.cuda()
        loc_gt = inputs['lm_loc_gt'].cuda()

        output_heat = model_HEAT(im)
        output_loc, output_vis = model_GHCU(output_heat)

        loss_dict = criterion(output_loc, output_vis, loc_gt, vis_gt)
        train_loss_dict = update_loss(loss_dict, train_loss_dict, loss_arr)
        loss_total = loss_dict['total']

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

    show_epoch_loss('train', train_loss_dict, len(train_loader), writer, epoch, tStart, loss_arr)

    # validation
    with torch.no_grad():
        for batch_idx, inputs in enumerate(validation_loader):

            # inputs = train_loader.next_batch()
            im = inputs['im_tensor'].cuda()
            [heat_gt, vis_gt] = inputs['labels']
            [heat_gt, vis_gt] = heat_gt.cuda(), vis_gt.cuda()
            loc_gt = inputs['lm_loc_gt'].cuda()

            output_heat = model_HEAT(im)
            output_loc, output_vis = model_GHCU(output_heat)

            loss_dict = criterion(output_loc, output_vis, loc_gt, vis_gt)
            valid_loss_dict = update_loss(loss_dict, valid_loss_dict, loss_arr)

    avg_total = show_epoch_loss('valid', valid_loss_dict, len(validation_loader), writer, epoch, tStart, loss_arr)

    scheduler.step(avg_total)

    train_loss_dict = {'loc': 0, 'vis': 0, 'total': 0}
    valid_loss_dict = {'loc': 0, 'vis': 0, 'total': 0}

    if (epoch+1) % 10 == 0:
        torch.save(model_HEAT.state_dict(), '../checkpoints/epoch_' + str(epoch+1) + '.pth')

