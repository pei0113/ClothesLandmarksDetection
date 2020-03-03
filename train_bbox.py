# -*- coding: UTF-8 -*-
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np
from time import time

from df_dataset_bbox import DFDatasets
from networks import HRNetFashionNet, DenseNet121Heat


def criterionHeat(out_heat, out_vis, gt_heat, gt_vis):
    x = out_heat - gt_heat              # (32, 6, 224, 224)
    x = torch.mul(x, x)
    x = torch.sum(torch.sum(x, 2), 2)   # (32, 6)
    x = x * torch.abs(gt_vis - out_vis) / (224*224)
    loss_heat = torch.sum(torch.sum(x, 0), 0) / (batch_size*6)   # (1)

    return loss_heat


use_gpu = True
lr = 0.001
batch_size = 32

validation_split = 0.2
shuffle_dataset = True
random_seed = 123

lm_txt = 'data/upper/train_list.txt'
bbox_txt = 'data/Anno/list_bbox.txt'
class_names = ["left collar", "right collar", "left sleeve", "right sleeve", "left hem", "right hem"]

# load data list
train_dataset = DFDatasets(lm_txt, bbox_txt)

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
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=10, sampler=train_sampler)
validation_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=10, sampler=validation_sampler)

# load FashionNet model
model = DenseNet121Heat()
print(model)

if use_gpu:
    model.cuda()

# train
criterionHeat = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler_2 = LambdaLR(optimizer, lr_lambda=[lambda epoch: 0.95 ** epoch])
scheduler = ReduceLROnPlateau(optimizer, mode='min', verbose=True)

total_train_loss = 0
total_val_loss = 0
for epoch in range(200):

    tStart = time()
    # scheduler_2.step()

    # train
    for batch_idx, inputs in enumerate(train_loader):

        # inputs = train_loader.next_batch()
        im = inputs['im_tensor'].cuda()
        [heat, vis] = inputs['labels']
        [heat, vis] = heat.cuda(), vis.cuda()

        output_heat = model(im)

        # loss = criterionHeat(output_heat, output_vis, heat, vis)
        loss = criterionHeat(output_heat, heat)
        total_train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # validation
    with torch.no_grad():
        for batch_idx, inputs in enumerate(validation_loader):

            # inputs = train_loader.next_batch()
            im = inputs['im_tensor'].cuda()
            [heat, vis] = inputs['labels']
            [heat, vis] = heat.cuda(), vis.cuda()

            output_heat, output_vis = model(im)

            loss = criterionHeat(output_heat, heat)
            # loss = criterionHeat(output_heat, output_vis, heat, vis)
            total_val_loss += loss

    avg_train_loss = total_train_loss/len(train_loader)
    avg_val_loss = total_val_loss/len(validation_loader)
    scheduler.step(avg_val_loss)

    print('==>>> **train** time: {:.3f}, epoch{}, train loss： {:.6f}, validation loss: {:.6f}'.format(time()-tStart, epoch+1, avg_train_loss, avg_val_loss))

    total_train_loss = 0
    total_val_loss = 0

    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), 'checkpoints/epoch_' + str(epoch+1) + '.pth')

