import os
import re
import cv2
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms


class DFDatasets(data.Dataset):
    def __init__(self, data_list):
        super(DFDatasets, self).__init__()

        # base setting
        self.data_list = data_list
        self.data_root = 'data'
        self.transform_x = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # load data & label list
        im_names = []
        labels = []
        with open(self.data_list, 'r') as f:
            for indx, line in enumerate(f.readlines()):
                # if indx == 500:
                #     break

                string = re.split('  | |\n', line)
                im_names.append(string[0])

                landmarks = string[3:21]
                labels.append(np.asarray([float(landmarks[x]) for x in range(0, len(landmarks))]))
        labels = np.asarray(labels)

        self.im_names = im_names
        self.labels = labels

    def __getitem__(self, index):
        # image processing
        im_name = self.im_names[index]
        im = Image.open(os.path.join(self.data_root, im_name))
        w, h = im.size
        im_tensor = self.transform_x(im)   # [0, 1]

        # label processing
        heats = np.zeros((6, 224, 224))
        labels = self.labels[index]
        vis = []
        x_gt = []
        y_gt = []
        count = 0
        for i in range(0, 18, 3):
            visible, x, y = labels[i:i+3]
            x_gt.append(x)
            y_gt.append(y)

            if visible == 2:        # cut-off
                visible_new = 0
            elif visible == 1:      # occlusion
                visible_new = 0.5
            elif visible == 0:      # visible
                visible_new = 1
            vis.append(visible_new)

            map = np.zeros((h, w))
            if visible == 2:
                # if landmark is cut-off map is all black
                continue
            else:
                x0 = 0 if x - 5 < 0 else x - 5
                y0 = 0 if y - 5 < 0 else y - 5
                x1 = w if x + 6 > w else x + 6
                y1 = h if y + 6 > h else y + 6
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
            'label_gt': label_test          # for testing
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