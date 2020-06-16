import os
import re
import cv2
import json
import numpy as np
from PIL import Image
from glob import glob

import torch
import torch.utils.data as data
from torchvision import transforms
from matplotlib import pyplot as plt
from gaussain import draw_heatmap, get_arraylike


class DFDatasets(data.Dataset):
    def __init__(self, image_path, bbox_path, DEBUG_MODE, data_root):
        super(DFDatasets, self).__init__()

        self.transform_x = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        im_names = []
        for im_name in glob(image_path + '*.jpg'):
            im_names.append(im_name)
        #     filename = im_name.split('/')[-1]
        #     bbox_json = os.path.join(bbox_path, filename.replace('jpg', 'json'))
        #     bbox_list = []
        #     with open(bbox_json, 'r') as json_file:
        #         json_data = json.loads(json_file.read())
        #         bbox_list.append(json_data['x'])
        #         bbox_list.append(json_data['y'])
        #         bbox_list.append(json_data['w'])
        #         bbox_list.append(json_data['h'])

        self.im_names = im_names

    def __getitem__(self, index):
        im_name = self.im_names[index]
        json_name = im_name.replace('jpg', 'json')
        json_name = json_name.replace('image', 'bbox')
        with open(json_name, 'r') as json_file:
            json_data = json.loads(json_file.read())
        bbox_x1 = json_data['x']
        bbox_y1 = json_data['y']
        bbox_x2 = json_data['w'] + bbox_x1
        bbox_y2 = json_data['h'] + bbox_y1

        im = Image.open(im_name)
        im_crop = im.crop((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
        # cv2.imshow('img', cv2.cvtColor(np.asarray(im_crop), cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        im_tensor = self.transform_x(im_crop)   # [0, 1]

        result = {
            'im_name': im_name,
            'im_tensor': im_tensor,
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
