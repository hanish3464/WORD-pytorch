import cv2
import numpy as np
import config
import preprocess
import random

#temp_img = config.test_images_folder_path + '0005-001.jpg'
#img = preprocess.loadImage(temp_img)
#temp_gt = config.test_ground_truth + 'res_0005-001.txt'
#arr_list, gt_len = preprocess.loadText(temp_gt)


def adjust_flipped_coordinate(img_shape, gt_list, status=None):
    flipped_gt = list()
    height, width, _ = img_shape
    adjust_wh_list = np.array([width,height]*4).reshape([4,2])

    for coord in gt_list:
        temp = (adjust_wh_list - np.array(coord))

        for i in range(4):
            if status == 'left-right': coord[i][0] = temp[i][0]
            elif status == 'top-bottom': coord[i][1] = temp[i][1]
        flipped_gt.append(coord)

    return flipped_gt

def flip(image, gt_list):
    opt = random.choice(['left-right', 'top-bottom'])
    image = image[:, ::-1, ::-1] if opt == 'left-right' else image[::-1, :, ::-1]
    image = np.ascontiguousarray(image, dtype=np.uint8)
    new_gt_list = adjust_flipped_coordinate(image.shape, gt_list, status=opt)

    while True:
        if not new_gt_list:
            break
        new_gt_element = new_gt_list.pop()
        poly = np.array(new_gt_element).astype(np.int32).reshape((-1)).reshape(-1, 2)
        cv2.polylines(image, [poly.reshape((-1, 1, 2))], True, color=(255, 0, 0), thickness=2)
        ptColor = (0, 255, 255)

    cv2.imwrite('/home/hanish/workspace/debug_image/debug_flip1.jpg', image)
    # cv2.imwrite('/home/hanish/workspace/debug_image/debug_flip2.jpg', image)

    return image, new_gt_list
#flip(img, arr_list)
