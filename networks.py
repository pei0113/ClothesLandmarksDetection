from torchvision.models import densenet121, vgg19

import torch.nn as nn
import torch.nn.functional as F

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

        base_model = model.features[:-1]
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


class DenseNet121Heat(nn.Module):
    def __init__(self):
        super(DenseNet121Heat, self).__init__()
        model = densenet121()
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
        y = self.avgpool(x)
        y = y.view(y.shape[0], -1)
        pose_vis = self.sigmoid(y)

        return pose_heat, pose_vis


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
        n11 = self.conv0(x)
        n11 = self.relu(self.bn64(n11))
        n11_ = self.conv64(n11)
        n11_ = self.relu(self.bn64(n11_))

        n21 = self.conv64(n11_)
        n21 = self.relu(self.bn64(n21))
        n22 = self.conv1(n11_)
        n22 = self.relu(self.bn128(n22))

        n21_ = self.conv64(n21)
        n21_ = self.bn64(n21_)
        n22_ = self.conv128(n22)
        n22_ = self.bn128(n22_)

        n21_fuse = n21_ + self.up1(n22_)
        n21_fuse = self.relu(n21_fuse)
        n22_fuse = self.bn128(self.conv1(n21_)) + n22_
        n22_fuse = self.relu(n22_fuse)

        n21_fuse_ = self.bn64(self.conv64(n21_fuse))
        n22_fuse_ = self.bn128(self.conv128(n22_fuse))

        n31 = n21_fuse_ + self.up1(n22_fuse_)
        n31 = self.relu(n31)
        n32 = self.conv1(n21_fuse_) + n22_fuse_
        n32 = self.relu(n32)
        n33 = self.bn256(self.conv2(self.bn128(self.conv1(n21_fuse_)))) + self.bn256(self.conv2(n22_fuse_))
        n33 = self.relu(n33)

        n31_ = self.conv64(n31)
        n32_ = self.conv128(n32)
        n33_ = self.conv256(n33)

        n31_fuse = n31_ + self.up1(n32_) + self.up1(self.up2(n33_))
        n31_fuse = self.relu(n31_fuse)
        # n32_fuse = self.relu(self.conv1(n31_)) + n32_ + self.up2(n33_)
        # n33_fuse = self.relu(self.conv2(self.relu(self.conv1(n31_)))) + self.relu(self.conv2(n32_)) + n33_

        pose_heat = self.sigmoid(self.heat(n31_fuse))

        x = self.avgpool(n31_fuse)
        x = x.view(x.shape[0], -1)
        pose_vis = self.sigmoid(self.vis(x))

        return pose_heat, pose_vis
