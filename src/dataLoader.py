import os
import re
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
            transforms.Resize(244),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform_y = transforms.Compose([
            transforms.ToTensor(),
        ])

        # load data & label list
        im_names = []
        labels = []
        with open(self.data_list, 'r') as f:
            for line in f.readlines():
                string = re.split('  | ', line)
                im_names.append(string[0])

                landmarks = string[3:-1]
                labels.append(np.asarray([int(landmarks[x]) for x in range(0, len(landmarks) - 1)]))
        labels = np.asarray(labels)

        self.im_names = im_names
        self.labels = labels

    def __getitem__(self, index):
        im_name = self.im_names[index]
        im = Image.open(os.path.join(self.data_root, im_name))
        im = self.transform_x(im)   # [0, 1]

        w, h = im.size
        labels = self.labels[index]
        for i in labels:
            if (i-1)%3 == 0:
                labels[i] = labels[i]/w
            elif (i-2)%3 == 0:
                labels[i] = labels[i]/h

        labels = self.transform_y(self.labels)  # [0, 1]

        result = {
            'im': im,
            'labels': labels
        }

        return result


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