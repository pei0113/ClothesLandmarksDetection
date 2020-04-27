import cv2

label = ['image_name', 'clothes_type', 'variation_type', 'landmark_visibility_1', 'landmark_location_x_1', 'landmark_location_y_1',
         'landmark_visibility_2', 'landmark_location_x_2', 'landmark_location_y_2', 'landmark_visibility_3', 'landmark_location_x_3',
         'landmark_location_y_3', 'landmark_visibility_4', 'landmark_location_x_4', 'landmark_location_y_4', 'landmark_visibility_5',
         'landmark_location_x_5', 'landmark_location_y_5', 'landmark_visibility_6', 'landmark_location_x_6', 'landmark_location_y_6',
         'landmark_visibility_7', 'landmark_location_x_7', 'landmark_location_y_7', 'landmark_visibility_8', 'landmark_location_x_8',
         'landmark_location_y_8']

landmarks = ["left collar", "right collar", "left sleeve", "right sleeve", "left hem", "right hem"]
color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]

list_path = 'data/upper/train_list_new2.txt'
# output_path = 'data/Anno/upper_clothes_landmarks_new.txt'
img_root = 'data/'

f = open(list_path, 'r')
lines = f.readlines()
total_number = len(lines)
n_visible = 0
n_occluded = 0
n_cutoff = 0

for nth, line in enumerate(lines):
    print('{} / {}'.format(nth, total_number))
    string = line.split('  ')

    # print(string[2])

    img = cv2.imread(img_root + string[0])

    for indx, point in enumerate(string[3:]):
        point = point.split(' ')
        visible, x, y = int(point[0]), int(point[1]), int(point[2])

        if visible == 0:
            result = 'visible'
            n_visible += 1
        elif visible == 1:
            result = 'occluded'
            n_occluded += 1
        else:
            result = 'cut-off'
            n_cutoff += 1
        cv2.circle(img, (x, y), 5, color[indx], -1)
        cv2.putText(img, result, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2, 2)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)

print("visible:{}, occluded:{}, cut-off:{}".format(n_visible, n_occluded, n_cutoff))