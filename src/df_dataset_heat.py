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
        labels = self.labels[index]
        vis = []
        x = []
        y = []
        for i in range(18):
            if (i-1) % 3 == 0:
                x.append(labels[i] * 224 / w)
            elif (i-2) % 3 == 0:
                y.append(labels[i] * 224 / h)
            else:
                v = labels[i]
                if v != 0:
                    v = 0.0
                else:
                    v = 1.0
                vis.append(v)
        x = torch.FloatTensor(np.asarray(x))        # [0, 1]
        y = torch.FloatTensor(np.asarray(y))        # [0, 1]
        vis = torch.FloatTensor(np.asarray(vis))    # [0, 1]

        labels = [vis, x, y]

        result = {
            'im_name': im_name,
            'im_tensor': im_tensor,
            'labels': labels,
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