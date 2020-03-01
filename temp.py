import shutil
import os
import cv2

f = open('data/Fashion_Landmark_Detection_Benchmark/upper/upper_clothes_landmarks.txt', 'r')
train_txt = open('data/Fashion_Landmark_Detection_Benchmark/upper/train_list.txt', 'r')
test_txt = open('data/Fashion_Landmark_Detection_Benchmark/upper/test_list.txt', 'r')

train_txt_new = open('data/Fashion_Landmark_Detection_Benchmark/upper/train_list_new.txt', 'w')
test_txt_new = open('data/Fashion_Landmark_Detection_Benchmark/upper/test_list_new.txt', 'w')

# root_dir = 'data/Fashion_Landmark_Detection_Benchmark/'
# train_dir = 'data/Fashion_Landmark_Detection_Benchmark/train/'
# test_dir = 'data/Fashion_Landmark_Detection_Benchmark/test/'

lines = f.readlines()
sep = int(len(lines)*0.8)

for line in train_txt.readlines():
    line.replace('img/', 'train/')
    train_txt_new.write(line)
    # string = line.split('  ')
    # shutil.move(os.path.join(root_dir, string[0]), train_dir)

for line in test_txt.readlines():
    line.replace('img/', 'test/')
    test_txt_new.write(line)
    # string = line.split('  ')
    # shutil.move(os.path.join(root_dir, string[0]), test_dir)