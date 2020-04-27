import cv2

label = ['image_name', 'clothes_type', 'variation_type', 'landmark_visibility_1', 'landmark_location_x_1', 'landmark_location_y_1',
         'landmark_visibility_2', 'landmark_location_x_2', 'landmark_location_y_2', 'landmark_visibility_3', 'landmark_location_x_3',
         'landmark_location_y_3', 'landmark_visibility_4', 'landmark_location_x_4', 'landmark_location_y_4', 'landmark_visibility_5',
         'landmark_location_x_5', 'landmark_location_y_5', 'landmark_visibility_6', 'landmark_location_x_6', 'landmark_location_y_6',
         'landmark_visibility_7', 'landmark_location_x_7', 'landmark_location_y_7', 'landmark_visibility_8', 'landmark_location_x_8',
         'landmark_location_y_8']

landmarks = ["left collar", "right collar", "left sleeve", "right sleeve", "left hem", "right hem"]
color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]

list_path = 'data/upper/test_list.txt'
output_path = 'data/upper/test_list1.txt'
img_root = 'data/'

f = open(list_path, 'r')
lines = f.readlines()

g = open(output_path, 'w')

for line in lines:
    string = line.split('  ')

    if string[2] == '1':
        g.write(line)

    # img = cv2.imread(img_root + string[0])

    # for indx, point in enumerate(string[3:]):
    #     point = point.split(' ')
    #     visible, x, y = int(point[0]), int(point[1]), int(point[2])
    #
    #     cv2.circle(img, (x, y), 3, color[indx], -1)
    #     cv2.imshow('img', img)
    #     cv2.waitKey(0)
