import shutil
import os
import cv2
import re
from PIL import Image
from math import sqrt, log

dmax = 100
edge_value = 0.01
print(sqrt(- pow(dmax, 2) / log(edge_value)))

# f = open('data/Fashion_Landmark_Detection_Benchmark/upper/upper_clothes_landmarks.txt', 'r')
# train_txt = open('data/Fashion_Landmark_Detection_Benchmark/upper/train_list.txt', 'r')
# test_txt = open('data/Fashion_Landmark_Detection_Benchmark/upper/test_list.txt', 'r')
#
# train_txt_new = open('data/Fashion_Landmark_Detection_Benchmark/upper/train_list_new.txt', 'w')
# test_txt_new = open('data/Fashion_Landmark_Detection_Benchmark/upper/test_list_new.txt', 'w')
#
# # root_dir = 'data/Fashion_Landmark_Detection_Benchmark/'
# # train_dir = 'data/Fashion_Landmark_Detection_Benchmark/train/'
# # test_dir = 'data/Fashion_Landmark_Detection_Benchmark/test/'
#
# lines = f.readlines()
# sep = int(len(lines)*0.8)
#
# for line in train_txt.readlines():
#     line.replace('img/', 'train/')
#     train_txt_new.write(line)
#     # string = line.split('  ')
#     # shutil.move(os.path.join(root_dir, string[0]), train_dir)
#
# for line in test_txt.readlines():
#     line.replace('img/', 'test/')
#     test_txt_new.write(line)
#     # string = line.split('  ')
#     # shutil.move(os.path.join(root_dir, string[0]), test_dir)

# ======================= show bbox label on image ======================

# lm_txt = open('data/upper/train_list.txt', 'r')
# bbox_txt = open('data/Anno/list_bbox.txt', 'r')
#
# lm_lines = lm_txt.readlines()
# bb_lines = bbox_txt.readlines()[2:]
#
# for lm_line in lm_lines:
#     filename = lm_line.split(' ')[0]
#     # img = cv2.imread('data/' + filename)
#     img = Image.open('data/' + filename)
#     nth_image = int(filename[10:18]) - 1
#     x1, y1, x2, y2 = bb_lines[nth_image].split(' ')[1:5]
#     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#     crop_img = img.crop((x1, y1, x2, y2))
#     crop_img.show()
#     canvas = img.copy()
#     canvas = cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
#     cv2.imshow('canvas', canvas)
#     cv2.waitKey(0)

# ======================= show the number of data that all landmarks are visible ======================
# txt_file = open('/media/ford/新增磁碟區/pei/deepFashion1-landmarks-detection/data/upper/train_list_new2.txt', 'r')
# lines = txt_file.readlines()
# all_vis = 0
#
# for line in lines:
#     n_visible = 0
#     string = re.split('  | |\n', line)
#     landmarks = string[3:21]
#
#     for i in range(0, 18, 3):
#         visible = landmarks[i]
#
#         if visible == '0':
#             n_visible += 1
#
#     if n_visible == 5:
#         all_vis += 1
#
# print('all landmarks is visible:{}'.format(all_vis))

# ======================= data balance ======================

# txt_file = open('/media/ford/新增磁碟區/pei/deepFashion1-landmarks-detection/data/upper/train_list_new.txt', 'r')
# new_txt = open('/media/ford/新增磁碟區/pei/deepFashion1-landmarks-detection/data/upper/train_list_new2.txt', 'w')
# lines = txt_file.readlines()
#
# for line in lines:
#     n_visible = 0
#     string = re.split('  | |\n', line)
#     landmarks = string[3:21]
#
#     for i in range(0, 18, 3):
#         visible = landmarks[i]
#
#         if visible == '0':            # if not all visible, break
#             n_visible += 1
#
#     if n_visible != 5:
#         new_txt.write(line)

