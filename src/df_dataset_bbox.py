import os
import re
import cv2
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms
from matplotlib import pyplot as plt
from gaussain import draw_heatmap, get_arraylike


class DFDatasets(data.Dataset):
    def __init__(self, lm_txt, bbox_txt, DEBUG_MODE):
        super(DFDatasets, self).__init__()

        # base setting
        self.lm_txt = lm_txt
        self.data_root = '../data'
        self.transform_x = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # load data & label list
        im_names = []
        landmarks = []
        bboxes = []
        bb_file = open(bbox_txt, 'r')
        bb_lines = bb_file.readlines()[2:]
        im_index = [re.split(' |/', line)[1] for line in bb_lines]

        with open(self.lm_txt, 'r') as f:
            if DEBUG_MODE:
                lines = f.readlines()[:500]
            else:
                lines = f.readlines()
            for indx, line in enumerate(lines):
                string = re.split('  | |\n', line)
                # img init
                im_names.append(string[0])
                # bbox init
                filename = string[0].split('/')[1]
                nth_img = im_index.index(filename)
                bbox = bb_lines[nth_img].split(' ')[1:5]
                bboxes.append(np.asarray([int(bbox[x]) for x in range(0, len(bbox))]))
                # landmark init
                lms = string[3:21]
                landmarks.append(np.asarray([float(lms[x]) for x in range(0, len(lms))]))
        bboxes = np.asarray(bboxes)
        landmarks = np.asarray(landmarks)

        self.im_names = im_names
        self.bboxes = bboxes
        self.landmarks = landmarks

    def __getitem__(self, index):
        # image processing
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = self.bboxes[index]
        im_name = self.im_names[index]
        im = Image.open(os.path.join(self.data_root, im_name))
        im_crop = im.crop((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
        # cv2.imshow('img', cv2.cvtColor(np.asarray(im_crop),cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        bbox_w, bbox_h = im_crop.size
        im_tensor = self.transform_x(im_crop)   # [0, 1]

        # label processing
        heats = np.zeros((6, 224, 224))
        landmarks = self.landmarks[index]
        x_gt = []           # for testing
        y_gt = []           # for testing
        x_gt_norm = []      # for evaluate
        y_gt_norm = []      # for evaluate
        conf_nocut_gt = []  # for training
        conf_vis_gt = []    # for training
        count = 0
        for i in range(0, 18, 3):
            visible, Lx, Ly = landmarks[i:i+3]
            Lx = Lx - bbox_x1
            Ly = Ly - bbox_y1
            x_gt.append(Lx)
            y_gt.append(Ly)
            Lx_norm = Lx / bbox_w
            Ly_norm = Ly / bbox_h
            x_gt_norm.append(Lx_norm)
            y_gt_norm.append(Ly_norm)

            # confidence label process
            if visible == 1:  # occlusion
                conf_nocut = 1
                conf_vis = 0
            elif visible == 0:  # visible
                conf_nocut = 1
                conf_vis = 1
            else:  # cut-off
                conf_nocut = 0
                conf_vis = 0
            conf_nocut_gt.append(conf_nocut)
            conf_vis_gt.append(conf_vis)

            map = np.zeros((bbox_h, bbox_w))
            if visible != 2:
                # Get heatmap(Method 1)
                array_like = get_arraylike(bbox_w, bbox_h)
                map = draw_heatmap(bbox_w, bbox_h, Lx, Ly, (bbox_w+bbox_h)//70, array_like)
                # Get heatmap(Method 2)
                # x0 = 0 if Lx - 5 < 0 else Lx - 5
                # y0 = 0 if Ly - 5 < 0 else Ly - 5
                # x1 = bbox_w if Lx + 6 > bbox_w else Lx + 6
                # y1 = bbox_h if Ly + 6 > bbox_h else Ly + 6
                # map[int(y0):int(y1), int(x0):int(x1)] = 1
            map = cv2.resize(map, (224, 224))
            # plt.imshow(map)
            # plt.show()

            heats[count, :, :] = map
            count += 1

        heats = torch.FloatTensor(heats)
        conf_nocut_gt = torch.FloatTensor(np.asarray(conf_nocut_gt))  # [0, 1]
        conf_vis_gt = torch.FloatTensor(np.asarray(conf_vis_gt))  # [0, 1]
        labels = [heats, conf_vis_gt]

        x_gt = torch.FloatTensor(np.asarray(x_gt))  # [0, 1]
        y_gt = torch.FloatTensor(np.asarray(y_gt))  # [0, 1]
        label_test = [conf_vis_gt, x_gt, y_gt]

        x_gt_norm = np.asarray(x_gt_norm)
        y_gt_norm = np.asarray(y_gt_norm)
        landmark_gt = [conf_vis_gt, x_gt_norm, y_gt_norm]

        result = {
            'im_name': im_name,
            'im_tensor': im_tensor,
            'labels': labels,               # for training
            'label_gt': label_test,         # for testing
            'landmark_gt': landmark_gt,     # for evaluate
            'bbox_tl': [bbox_x1, bbox_y1, bbox_x2, bbox_y2]       # for testing
        }

        return result

    def __len__(self):
        return len(self.im_names)


class DFDataLoader(object):
    def __init__(self, batch_size, dataset):
        super(DFDataLoader, self).__init__()

        self.data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            pin_memory=True)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()
        return batch
