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

sys.path.append(os.path.abspath(".."))

from df_dataset_bbox import DFDatasets
from networks import DenseNet121Heat, HRNetFashionNet
from utils import update_loss, show_epoch_loss, set_random_seed


def criterionHeat(out_heat, gt_heat, gt_vis):
    criterion = nn.MSELoss()
    loss_heat = criterion(out_heat, gt_heat)
    # batch = out_heat.shape[0]
    # x = out_heat - gt_heat              # (32, 6, 224, 224)
    # x = torch.mul(x, x)
    # x = torch.sum(torch.sum(x, 2), 2) / (224*224)   # (32, 6)
    # # x = x * gt_vis
    # loss_heat = torch.sum(torch.sum(x, 0), 0) / (batch*6)   # (1)
    loss_dict = {'heat': loss_heat}

    return loss_dict


DEBUG_MODE = True
num_worker = 0
use_gpu = True
lr = 0.001
batch_size = 20
validation_split = 0.2
shuffle_dataset = True
set_random_seed(2020)

# tensor board
writer = SummaryWriter()
root = '../'
lm_txt = root + '/data/upper/train_list.txt'
bbox_txt = root + '/data/Anno/list_bbox.txt'
class_names = ["left collar", "right collar", "left sleeve", "right sleeve", "left hem", "right hem"]

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

# load FashionNet model
model = HRNetFashionNet()

if use_gpu:
    model.cuda()

# train
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', verbose=True)

train_loss_dict = {'heat': 0}
valid_loss_dict = {'heat': 0}
loss_arr = ['heat']
for epoch in range(200):

    tStart = time()
    # scheduler_2.step()

    # train
    for batch_idx, inputs in enumerate(train_loader):

        im = inputs['im_tensor'].cuda()
        [heat, vis] = inputs['labels']
        [heat, vis] = heat.cuda(), vis.cuda()

        output_heat = model(im)

        loss_dict = criterionHeat(output_heat, heat, vis)
        train_loss_dict = update_loss(loss_dict, train_loss_dict, loss_arr)
        loss_total = loss_dict['heat']

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

    show_epoch_loss('train', train_loss_dict, len(train_loader), writer, epoch, tStart, loss_arr)

    # validation
    with torch.no_grad():
        for batch_idx, inputs in enumerate(validation_loader):

            # inputs = train_loader.next_batch()
            im = inputs['im_tensor'].cuda()
            [heat, vis] = inputs['labels']
            [heat, vis] = heat.cuda(), vis.cuda()

            output_heat = model(im)

            loss_dict = criterionHeat(output_heat, heat, vis)
            valid_loss_dict = update_loss(loss_dict, valid_loss_dict, loss_arr)

    avg_total = show_epoch_loss('valid', valid_loss_dict, len(validation_loader), writer, epoch, tStart, loss_arr)

    scheduler.step(avg_total)

    train_loss_dict = {'heat': 0}
    valid_loss_dict = {'heat': 0}

    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), root + 'checkpoints/epoch_' + str(epoch+1) + '.pth')

