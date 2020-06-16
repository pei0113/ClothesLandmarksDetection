from torchvision.models import densenet121, vgg19, inception_v3
from unet_parts import down, inconv, outconv, up

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

ROI_POOL_SIZE = (3, 3)
N_LANDMARKS = 6


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Vgg16FashionNet(nn.Module):
    def __init__(self):
        super(Vgg16FashionNet, self).__init__()
        model = vgg19()

        features = list(model.features.children())
        self.conv4 = nn.Sequential(*features[:-8])  # the paper implements DF with features taken from conv4 from vgg16
        self.conv5_pose = nn.Sequential(*features[-8:])
        self.conv5_global = nn.Sequential(*features[-8:])

        self.fc6_global = nn.Linear(in_features=512 * 7 * 7, out_features=1024 * 4)
        self.fc6_local = nn.Linear(in_features=512 * N_LANDMARKS * ROI_POOL_SIZE[0] * ROI_POOL_SIZE[1],
                                   out_features=1024)
        self.fc6_pose = nn.Linear(in_features=512 * 7 * 7, out_features=1024)
        self.fc7_pose = nn.Linear(in_features=1024, out_features=1024)
        self.loc = nn.Linear(in_features=1024, out_features=12)
        self.vis = nn.Linear(in_features=1024, out_features=6)

        self.flatten = Flatten()

    def forward(self, x):
        base_features = self.conv4(x)
        pose = self.flatten(self.conv5_pose(base_features))

        pose = F.leaky_relu(self.fc6_pose(pose))
        pose = F.leaky_relu(self.fc7_pose(pose))

        pose_loc = F.leaky_relu(self.loc(pose))
        pose_vis = F.sigmoid(self.vis(pose))

        return pose_loc, pose_vis


class DenseNet121FashionNet(nn.Module):
    def __init__(self):
        super(DenseNet121FashionNet, self).__init__()
        model = densenet121()

        base_model = model.features
        self.base_model = base_model
        self.fc_last = nn.Linear(in_features=1024 * 7 * 7, out_features=1024)
        self.loc = nn.Linear(in_features=1024, out_features=12)
        self.vis = nn.Linear(in_features=1024, out_features=6)
        self.flatten = Flatten()

        # num_features = model.classifier.in_features
        # features = nn.Linear(num_features, output_classes)
        # model.classifier = features

    def forward(self, x):
        based_feature = self.flatten(self.base_model(x))
        pose = F.leaky_relu(self.fc_last(based_feature))

        pose_loc = F.leaky_relu(self.loc(pose))
        pose_vis = F.sigmoid(self.vis(pose))

        return pose_loc, pose_vis


class LVNet(nn.Module):
    def __init__(self):
        super(LVNet, self).__init__()
        model = densenet121()

        base_model = model.features
        self.base_model = base_model
        self.fc1 = nn.Linear(in_features=1024 * 7 * 7, out_features=1024)
        self.coord = nn.Linear(in_features=1024, out_features=12)
        self.confNOCUT = nn.Linear(in_features=1024, out_features=6)
        self.confVIS = nn.Linear(in_features=1024, out_features=6)
        self.flatten = Flatten()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()

        # num_features = model.classifier.in_features
        # features = nn.Linear(num_features, output_classes)
        # model.classifier = features

    def forward(self, x):
        based_feature = self.base_model(x)
        x = self.flatten(based_feature)
        x = self.fc1(x)
        x = self.relu(x)
        coord = self.leakyrelu(self.coord(x))
        conf_nocut = self.sigmoid(self.confNOCUT(x))
        conf_vis = self.sigmoid(self.confVIS(x))
        return coord, conf_vis


class DenseNet121Heat(nn.Module):
    def __init__(self):
        super(DenseNet121Heat, self).__init__()
        model = densenet121(pretrained=True)
        base_model = model.features[:-6]            # [128*28*28]

        self.base_model = base_model
        self.up1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=1),
            nn.BatchNorm2d(64),
            nn.Upsample(size=(112, 112), mode='nearest'),
            )
        self.up2 = nn.Sequential(
            nn.Conv2d(64, 6, kernel_size=1, bias=1),
            nn.BatchNorm2d(6),
            nn.Upsample(size=(224, 224), mode='nearest'))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        based_feature = self.base_model(x)
        x = self.up1(based_feature)
        x = self.up2(x)
        pose_heat = self.sigmoid(x)
        # y = self.avgpool(x)
        # y = y.view(y.shape[0], -1)
        # pose_vis = self.sigmoid(y)

        return pose_heat


class DenseNet121Visible(nn.Module):
    def __init__(self):
        super(DenseNet121Visible, self).__init__()
        model = densenet121()
        base_model = model.features[:-6]

        self.base_model = base_model
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout2d(0.2)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 12)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        based_feature = self.base_model(x)          # [128*28*28]
        x = self.avgpool(based_feature)             # [128*1*1]
        x = self.relu(x)
        x = self.drop(x)
        x = x.view(x.shape[0], -1)                  # [128]
        x = self.fc1(x)                             # [64]
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)                             # [12]
        pose_vis = self.sigmoid(x)
        return pose_vis


class Conv3x3_Bn_Relu(nn.Sequential):
    def __init__(self, in_num, out_num):
        super(Conv3x3_Bn_Relu, self).__init__()
        self.add_module('conv', nn.Conv2d(in_num, out_num, kernel_size=3, stride=2))
        self.add_module('norm', nn.BatchNorm2d(out_num))
        self.add_module('relu', nn.ReLU(inplace=True))

class ghcu(nn.Module):
    def __init__(self, in_channel):
        super(ghcu, self).__init__()
        self.conv1 = Conv3x3_Bn_Relu(in_channel, 64)
        self.conv2 = Conv3x3_Bn_Relu(64, 64)
        self.conv3 = Conv3x3_Bn_Relu(64, 32)
        self.conv4 = Conv3x3_Bn_Relu(32, 32)
        self.conv5 = Conv3x3_Bn_Relu(32, 16)
        self.conv6 = Conv3x3_Bn_Relu(16, 16)
        self.flatten = Flatten()
        self.drop = nn.Dropout2d(0.2)
        self.loc = nn.Linear(16 * 2 * 2, in_channel*2)
        self.vis = nn.Linear(16 * 2 * 2, in_channel)
        self.leakyRelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.drop(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.drop(x)
        x = self.flatten(x)
        lm_loc = self.leakyRelu(self.loc(x))
        lm_vis = self.sigmoid(self.vis(x))
        return lm_loc, lm_vis


class ghcu2(nn.Module):
    def __init__(self, in_channel):
        super(ghcu2, self).__init__()
        self.conv1 = Conv3x3_Bn_Relu(in_channel, 64)
        self.conv2 = Conv3x3_Bn_Relu(64, 64)
        self.conv3 = Conv3x3_Bn_Relu(64, 32)
        self.conv4 = Conv3x3_Bn_Relu(32, 32)
        self.conv5 = Conv3x3_Bn_Relu(32, 16)
        self.conv6 = Conv3x3_Bn_Relu(16, 16)
        self.flatten = Flatten()
        self.drop = nn.Dropout2d(0.2)
        self.loc = nn.Linear(16 * 2 * 2, in_channel*2)
        self.leakyRelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.drop(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.drop(x)
        x = self.flatten(x)
        lm_loc = self.leakyRelu(self.loc(x))
        return lm_loc


class GHCU(nn.Module):
    def __init__(self):
        super(GHCU, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, stride=2)
        self.conv6 = nn.Conv2d(16, 16, kernel_size=3, stride=2)
        self.flatten = Flatten()
        self.drop = nn.Dropout2d(0.2)
        self.loc = nn.Linear(16*2*2, 12)
        self.vis = nn.Linear(16*2*2, 6)
        self.relu = nn.ReLU()
        self.leakyRelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.flatten(x)
        lm_loc = self.leakyRelu(self.loc(x))
        lm_vis = self.sigmoid(self.vis(x))
        return lm_loc, lm_vis


class GHCU_visible(nn.Module):
    def __init__(self):
        super(GHCU_visible, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, stride=2)
        self.conv6 = nn.Conv2d(16, 16, kernel_size=3, stride=2)
        self.flatten = Flatten()
        self.drop = nn.Dropout2d(0.2)
        self.loc = nn.Linear(16*2*2, 12)
        self.vis = nn.Linear(16*2*2, 6)
        self.relu = nn.ReLU()
        self.leakyRelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.flatten(x)
        lm_vis = self.sigmoid(self.vis(x))
        return lm_vis


class HeatLVNet(nn.Module):
    def __init__(self):
        super(HeatLVNet, self).__init__()
        model = densenet121(pretrained=True)
        base_model = model.features[:-6]            # [128*28*28]

        self.base_model = base_model
        self.up1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=1),
            nn.BatchNorm2d(64),
            nn.Upsample(size=(112, 112), mode='nearest'),
            )
        self.up2 = nn.Sequential(
            nn.Conv2d(64, 6, kernel_size=1, bias=1),
            nn.BatchNorm2d(6),
            nn.Upsample(size=(224, 224), mode='nearest'))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.vis = nn.Linear(128, 6)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        based_feature = self.base_model(x)      # (128, 28, 28)
        y = self.avgpool(based_feature)         # (128, 1, 1)
        y = y.view(y.shape[0], -1)              # (128)
        y = self.vis(y)
        lm_vis = self.sigmoid(y)
        x = self.up1(based_feature)             # (64, 112, 112)
        x = self.up2(x)                         # (6, 224, 224)
        lm_heat = self.sigmoid(x)
        return lm_heat, lm_vis


class LVUNet(nn.Module):
    def __init__(self):
        super(LVUNet, self).__init__()
        self.inc = inconv(3, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.vis = nn.Linear(512, 6)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, 6)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)         # (512, 14, 14)
        y = self.avgpool(x5)        # (512, 1, 1)
        y = y.view(y.shape[0], -1)  # (512)
        y = self.vis(y)             # (6)
        vis = self.sigmoid(y)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        heat = self.sigmoid(x)
        return heat, vis


class LVUNet2(nn.Module):
    def __init__(self):
        super(LVUNet2, self).__init__()
        self.inc = inconv(3, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 256)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.vis = nn.Linear(256, 6)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, 6)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)         # (256, 28, 28)
        f = self.inc(x)
        f = self.down1(f)
        f = self.down2(f)
        f = self.down3(f)           # (256, 28, 28)
        y = torch.cat([x4, f], 2)   # (512, 28, 28)
        y = self.avgpool(y)         # (512, 1, 1)
        y = y.view(y.shape[0], -1)  # (512)
        y = self.vis(y)             # (6)
        vis = self.sigmoid(y)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        heat = self.sigmoid(x)
        return heat, vis


class LVUNet3(nn.Module):
    def __init__(self):
        super(LVUNet3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.inc = inconv(64, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.vis = nn.Linear(512, 6)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, 6)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feature = self.features(x)

        x1 = self.inc(feature)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)         # (512, 3, 3)

        y = self.avgpool(x5)        # (512, 1, 1)
        y = y.view(y.shape[0], -1)  # (512)
        y = self.vis(y)             # (6)
        vis1 = self.sigmoid(y)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)         # (64, 56, 56)
        # unet2
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)         # (512, 3, 3)

        y = self.avgpool(x5)        # (512, 1, 1)
        y = y.view(y.shape[0], -1)  # (512)
        y = self.vis(y)             # (6)
        vis2 = self.sigmoid(y)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)         # (64, 56, 56)
        x = self.outc(x)
        heat = self.sigmoid(x)
        return heat, torch.add(vis1, vis2) / 2


class LVUNet4(nn.Module):
    def __init__(self):
        super(LVUNet4, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.inc = inconv(64, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 128)
        self.down3 = down(128, 128)
        self.down4 = down(128, 128)
        self.vis = nn.Linear(3*3*128, 6)
        self.up1 = up(256, 128)
        self.up2 = up(256, 128)
        self.up3 = up(256, 128)
        self.up4 = up(192, 64)
        self.outc = outconv(64, 6)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feature = self.features(x)      # (64, 56, 56)
        x1 = self.inc(feature)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)             # (128, 3, 3)
        y = x5.view(x5.shape[0], -1)    # (128*3*3)
        y = self.vis(y)                 # (6)
        vis = self.sigmoid(y)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        heat = self.sigmoid(x)
        return heat, vis


class LVUNet5(nn.Module):
    def __init__(self):
        super(LVUNet5, self).__init__()
        self.inc = inconv(3, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.vis = nn.Linear(512, 6)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, 6)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)         # (512, 14, 14)
        y = self.avgpool(x5)        # (512, 1, 1)
        y = y.view(y.shape[0], -1)  # (512)
        y = self.vis(y)             # (6)
        vis = self.sigmoid(y)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        heat = self.tanh(x)
        return heat


class LVUNet_GHCU(nn.Module):
    def __init__(self, n_keypoint):
        super(LVUNet_GHCU, self).__init__()
        self.inc = inconv(3, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_keypoint)
        self.ghcu = ghcu(n_keypoint)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        heat = self.tanh(x)
        lm_loc, lm_vis = self.ghcu(x)
        return heat, lm_loc, lm_vis


class LVUNet_GHCU2(nn.Module):
    def __init__(self, n_keypoint):
        super(LVUNet_GHCU2, self).__init__()
        self.inc = inconv(3, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_keypoint)
        self.ghcu = ghcu(n_keypoint)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        heat = self.tanh(x)
        lm_loc, lm_vis = self.ghcu(heat)
        return heat, lm_loc, lm_vis


class LVUNet_GHCU3(nn.Module):
    def __init__(self, n_keypoint):
        super(LVUNet_GHCU3, self).__init__()
        self.inc = inconv(3, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_keypoint)
        self.ghcu = ghcu2(n_keypoint)           # only output loc
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        heat = self.tanh(x)
        lm_loc = self.ghcu(heat)
        return heat, lm_loc


class HRNetFashionNet(nn.Module):
    def __init__(self):
        super(HRNetFashionNet, self).__init__()

        self.conv0 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv64 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv128 = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.conv256 = nn.Conv2d(256, 256, kernel_size=1, bias=False)
        self.bn64 = nn.BatchNorm2d(64)
        self.bn128 = nn.BatchNorm2d(128)
        self.bn256 = nn.BatchNorm2d(256)

        self.up1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=1),
            nn.BatchNorm2d(64),
            nn.Upsample(size=(224, 224), mode='nearest')
            )
        self.up2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=1),
            nn.BatchNorm2d(128),
            nn.Upsample(size=(112, 112), mode='nearest')
            )

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.heat = nn.Sequential(
            nn.Conv2d(64, 6, kernel_size=1, bias=False),
            nn.BatchNorm2d(6))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.vis = nn.Linear(64, 6)

    def forward(self, x):
        n11 = self.conv0(x)                 # (64, 224, 224)
        n11 = self.relu(self.bn64(n11))
        n11_ = self.conv64(n11)
        n11_ = self.relu(self.bn64(n11_))

        n21 = self.conv64(n11_)             # (64, 224, 224)
        n21 = self.relu(self.bn64(n21))
        n22 = self.conv1(n11_)              # (128, 112, 112)
        n22 = self.relu(self.bn128(n22))

        n21_ = self.conv64(n21)             # (64, 224, 224)
        n21_ = self.bn64(n21_)
        n22_ = self.conv128(n22)            # (128, 112, 112)
        n22_ = self.bn128(n22_)

        n21_fuse = n21_ + self.up1(n22_)
        n21_fuse = self.relu(n21_fuse)                  # (64, 224, 224)
        n22_fuse = self.bn128(self.conv1(n21_)) + n22_
        n22_fuse = self.relu(n22_fuse)                  # (128, 112, 112)

        n21_fuse_ = self.bn64(self.conv64(n21_fuse))
        n22_fuse_ = self.bn128(self.conv128(n22_fuse))

        n31 = n21_fuse_ + self.up1(n22_fuse_)           # (64, 224, 224)
        n31 = self.relu(n31)
        n32 = self.conv1(n21_fuse_) + n22_fuse_
        n32 = self.relu(n32)                            # (128, 112, 112)
        n33 = self.bn256(self.conv2(self.bn128(self.conv1(n21_fuse_)))) + self.bn256(self.conv2(n22_fuse_))
        n33 = self.relu(n33)                            # (256, 56, 56)

        n31_ = self.conv64(n31)             # (64, 224, 224)
        n32_ = self.conv128(n32)            # (128, 112, 112)
        n33_ = self.conv256(n33)            # (256, 56, 56)

        n31_fuse = n31_ + self.up1(n32_) + self.up1(self.up2(n33_))
        n31_fuse = self.relu(n31_fuse)
        # n32_fuse = self.relu(self.conv1(n31_)) + n32_ + self.up2(n33_)
        # n33_fuse = self.relu(self.conv2(self.relu(self.conv1(n31_)))) + self.relu(self.conv2(n32_)) + n33_

        pose_heat = self.sigmoid(self.heat(n31_fuse))

        # x = self.avgpool(n31_fuse)
        # x = x.view(x.shape[0], -1)
        # pose_vis = self.sigmoid(self.vis(x))

        return pose_heat
