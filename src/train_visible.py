# -*- coding: UTF-8 -*-
import torch
import torch.optim as optim
import torch.nn as nn
from termcolor import cprint
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

import numpy as np
import cv2
from time import time
from tensorboardX import SummaryWriter

from df_dataset_visible import DFDatasets
from networks import LVNet, DenseNet121FashionNet
from loss import criterionFOCAL, criterionBCE, criterionFOCAL_vis, criterionBCE_vis
from utils import update_loss, show_epoch_loss, set_random_seed


def criterionLV(output, x_gt, y_gt, conf_nocut, conf_vis):
    out_coord, out_conf_nocut, out_conf_vis = output
    out_x, out_y = torch.split(out_coord, 6, dim=1)

    # if no cut-off
    loss_x = torch.abs(out_x - x_gt)
    loss_y = torch.abs(out_y - y_gt)
    loss_coord = torch.sum(conf_nocut * (loss_x + loss_y)) / torch.sum(conf_nocut)
    loss_coord = LambdaCOOR * loss_coord

    # confidence no cut
    loss_nocut = MAE(out_conf_nocut, conf_nocut, conf_nocut)
    loss_nocut = LambdaNOCUT * loss_nocut

    # if cut-off
    conf_cut = torch.ones(6).cuda() - conf_nocut
    loss_cut = MAE(out_conf_nocut, conf_nocut, conf_cut)
    loss_cut = LambdaCUT * loss_cut

    # if no cut-off and visible
    loss_vis = MAE(out_conf_vis, conf_vis, conf_vis)
    loss_vis = LambdaVIS * loss_vis

    # if no cut-off and occluded
    conf_occ = torch.ones(6).cuda() - conf_vis
    loss_occ = MAE(out_conf_vis, conf_vis, conf_occ)
    loss_occ = LambdaOCC * loss_occ

    loss_total = loss_coord + loss_nocut + loss_cut + loss_vis + loss_occ

    loss_dict = {'total': loss_total,
                 'coord': loss_coord,
                 'nocut': loss_nocut,
                 'cut': loss_cut,
                 'vis': loss_vis,
                 'occ': loss_occ}
    return loss_dict


DEBUG_MODE = False
num_worker = 10
use_gpu = True
lr = 0.001
batch_size = 32
validation_split = 0.2
shuffle_dataset = True
set_random_seed(2020)
LambdaCOOR = 20
LambdaNOCUT = 1
LambdaCUT = 11.6
LambdaVIS = 0.25
LambdaOCC = 0.75

# tensor board
writer = SummaryWriter()

lm_txt = 'data/upper/train_list.txt'
bbox_txt = 'data/Anno/list_bbox.txt'
class_names = ["left collar", "right collar", "left sleeve", "right sleeve", "left hem", "right hem"]

# load data list
train_dataset = DFDatasets(lm_txt, bbox_txt, DEBUG_MODE)

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
model = LVNet()
# model = DenseNet121FashionNet()

if use_gpu:
    model.cuda()

# train
# criterionVisible = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler_2 = LambdaLR(optimizer, lr_lambda=[lambda epoch: 0.95 ** epoch])
scheduler = ReduceLROnPlateau(optimizer, mode='min', verbose=True)

# train_loss_dict = {'total': 0, 'coord': 0, 'nocut': 0, 'cut': 0, 'vis': 0, 'occ': 0}
# valid_loss_dict = {'total': 0, 'coord': 0, 'nocut': 0, 'cut': 0, 'vis': 0, 'occ': 0}
# train_loss_dict = {'total': 0, 'coord': 0, 'nocut': 0, 'vis': 0}
# valid_loss_dict = {'total': 0, 'coord': 0, 'nocut': 0, 'vis': 0}
train_loss_dict = {'total': 0, 'coord': 0, 'vis': 0}
valid_loss_dict = {'total': 0, 'coord': 0, 'vis': 0}
loss_arr = ['total', 'coord', 'vis']

for epoch in range(200):
    # train
    # model.train()
    tStart = time()
    for batch_idx, inputs in enumerate(train_loader):
        im = inputs['im_tensor'].cuda()
        # x_gt, y_gt, conf_nocut, conf_vis = inputs['label_gt']
        # x_gt, y_gt, conf_nocut, conf_vis = x_gt.cuda(), y_gt.cuda(), conf_nocut.cuda(), conf_vis.cuda()
        x_gt, y_gt, conf_vis = inputs['label_gt']
        x_gt, y_gt, conf_vis = x_gt.cuda(), y_gt.cuda(), conf_vis.cuda()

        output = model(im)

        # loss_dict = criterionFOCAL(output, x_gt, y_gt, conf_nocut, conf_vis, r=1)
        loss_dict = criterionBCE_vis(output, x_gt, y_gt, conf_vis)
        train_loss_dict = update_loss(loss_dict, train_loss_dict, loss_arr)

        loss_total = loss_dict['total']
        loss_total.requires_grad_()
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

    show_epoch_loss('train', train_loss_dict, len(train_loader), writer, epoch, tStart, loss_arr)

    # validation
    # model.eval()
    with torch.no_grad():
        tStart = time()
        for batch_idx, inputs in enumerate(validation_loader):
            im = inputs['im_tensor'].cuda()
            # x_gt, y_gt, conf_nocut, conf_vis = inputs['label_gt']
            # x_gt, y_gt, conf_nocut, conf_vis = x_gt.cuda(), y_gt.cuda(), conf_nocut.cuda(), conf_vis.cuda()
            x_gt, y_gt, conf_vis = inputs['label_gt']
            x_gt, y_gt, conf_vis = x_gt.cuda(), y_gt.cuda(), conf_vis.cuda()

            output = model(im)

            # loss_dict = criterionFOCAL(output, x_gt, y_gt, conf_nocut, conf_vis, r=1))
            loss_dict = criterionBCE_vis(output, x_gt, y_gt, conf_vis)
            valid_loss_dict = update_loss(loss_dict, valid_loss_dict, loss_arr)

    avg_total = show_epoch_loss('valid', valid_loss_dict, len(validation_loader), writer, epoch, tStart, loss_arr)

    scheduler.step(avg_total)

    # train_loss_dict = {'total': 0, 'coord': 0, 'nocut': 0, 'cut': 0, 'vis': 0, 'occ': 0}
    # valid_loss_dict = {'total': 0, 'coord': 0, 'nocut': 0, 'cut': 0, 'vis': 0, 'occ': 0}
    # train_loss_dict = {'total': 0, 'coord': 0, 'nocut': 0, 'vis': 0}
    # valid_loss_dict = {'total': 0, 'coord': 0, 'nocut': 0, 'vis': 0}
    train_loss_dict = {'total': 0, 'coord': 0, 'vis': 0}
    valid_loss_dict = {'total': 0, 'coord': 0, 'vis': 0}

    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), 'checkpoints/epoch_' + str(epoch+1) + '.pth')

