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

from df_dataset_heat import DFDatasets
from networks import Vgg16FashionNet, DenseNet121FashionNet


use_gpu = True
output_classes = 18
lr = 0.0001
batch_size = 16

validation_split = 0.2
shuffle_dataset = True
random_seed = 123


def criterion(out_x, x, out_y, y, vis):
    loss_x = torch.abs(out_x - x)
    loss_y = torch.abs(out_y - y)
    loss = vis * (loss_x + loss_y)
    loss = torch.mul(loss, loss)
    loss = torch.sum(loss)
    return loss


def criterion2(vis, x, y, output):
    x /= 224
    y /= 224
    out_x, out_y = torch.split(output[0], 6, 1)
    out_vis = output[1]

    vis_criterion = nn.MSELoss()
    vis_loss = vis_criterion(out_vis, vis)

    loss_x = torch.abs(out_x - x)
    loss_y = torch.abs(out_y - y)
    loss = vis * (loss_x + loss_y)
    # loss = torch.mul(loss, loss)
    loss = (torch.sum(loss)/(batch_size*6) + vis_loss)/2
    return loss


train_txt = 'data/upper/train_list.txt'
test_txt = 'data/upper/test_list.txt'
class_names = ["left collar", "right collar", "left sleeve", "right sleeve", "left hem", "right hem"]

# load data list
train_dataset = DFDatasets(train_txt)

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
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=1, sampler=train_sampler)
validation_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=1, sampler=validation_sampler)

# load FashionNet model
model = DenseNet121FashionNet()
print(model)

if use_gpu:
    model.cuda()

# train
criterionLM = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler_2 = LambdaLR(optimizer, lr_lambda=[lambda epoch: 0.95 ** epoch])
scheduler = ReduceLROnPlateau(optimizer, mode='min', verbose=True)

total_train_loss = 0
total_val_loss = 0
for epoch in range(100):

    tStart = time()
    # scheduler_2.step()

    # train
    for batch_idx, inputs in enumerate(train_loader):

        # inputs = train_loader.next_batch()
        im = inputs['im_tensor'].cuda()
        [vis, x, y] = inputs['labels']
        [vis, x, y] = vis.cuda(), x.cuda(), y.cuda()

        output = model(im)

        loss = criterion2(vis, x, y, output)
        total_train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # validation
    with torch.no_grad():
        for batch_idx, inputs in enumerate(validation_loader):

            # inputs = train_loader.next_batch()
            im = inputs['im_tensor'].cuda()
            [vis, x, y] = inputs['labels']
            [vis, x, y] = vis.cuda(), x.cuda(), y.cuda()

            output = model(im)

            loss = criterion2(vis, x, y, output)
            total_val_loss += loss

    avg_train_loss = total_train_loss/len(train_loader)
    avg_val_loss = total_val_loss/len(validation_loader)
    scheduler.step(avg_val_loss)

    print('==>>> **train** time: {:.3f}, epoch{}, train loss： {:.6f}, validation loss: {:.6f}'.format(time()-tStart, epoch+1, avg_train_loss, avg_val_loss))

    total_train_loss = 0
    total_val_loss = 0

    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), 'checkpoints/epoch_' + str(epoch+1) + '.pth')

