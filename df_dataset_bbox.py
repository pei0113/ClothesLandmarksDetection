import os
import re
import cv2
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms


class DFDatasets(data.Dataset):
    def __init__(self, lm_txt, bbox_txt):
        super(DFDatasets, self).__init__()

        # base setting
        self.lm_txt = lm_txt
        self.data_root = 'data'
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
            for indx, line in enumerate(f.readlines()):
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
        bbox_w, bbox_h = im_crop.size
        im_tensor = self.transform_x(im_crop)   # [0, 1]

        # label processing
        heats = np.zeros((6, 224, 224))
        landmarks = self.landmarks[index]
        vis = []
        x_gt = []       # for testing
        y_gt = []       # for testing
        count = 0
        for i in range(0, 18, 3):
            visible, Lx, Ly = landmarks[i:i+3]
            Lx = Lx - bbox_x1
            Ly = Ly - bbox_y1
            x_gt.append(Lx)
            y_gt.append(Ly)

            if visible == 2:        # cut-off
                visible_new = 0
            elif visible == 1:      # occlusion
                visible_new = 0.5
            elif visible == 0:      # visible
                visible_new = 1
            vis.append(visible_new)

            map = np.zeros((bbox_h, bbox_w))
            if visible == 2:
                # if landmark is cut-off map is all black
                continue
            else:
                x0 = 0 if Lx - 5 < 0 else Lx - 5
                y0 = 0 if Ly - 5 < 0 else Ly - 5
                x1 = bbox_w if Lx + 6 > bbox_w else Lx + 6
                y1 = bbox_h if Ly + 6 > bbox_h else Ly + 6
                map[int(y0):int(y1), int(x0):int(x1)] = 1
            map = cv2.resize(map, (224, 224))
            heats[count, :, :] = map
            count += 1

        heats = torch.FloatTensor(heats)
        vis = torch.FloatTensor(np.asarray(vis))    # [0, 0.5, 1]
        labels = [heats, vis]

        x_gt = torch.FloatTensor(np.asarray(x_gt))  # [0, 1]
        y_gt = torch.FloatTensor(np.asarray(y_gt))  # [0, 1]
        label_test = [vis, x_gt, y_gt]

        result = {
            'im_name': im_name,
            'im_tensor': im_tensor,
            'labels': labels,               # for training
            'label_gt': label_test,         # for testing
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