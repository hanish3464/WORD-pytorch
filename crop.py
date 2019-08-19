import cv2
import random
import config
import preprocess
import numpy as np

#temp_img = config.test_images_folder_path + '0005-001.jpg'
#img = preprocess.loadImage(temp_img)
#temp_gt = config.test_ground_truth + '15.txt'
#arr_list, gt_len = preprocess.loadText(temp_gt)


def adjust_cropped_coordinate(lt, rt, lb, rb, coord):
    # cropped image apex list of boolean / left_top, right_top, left_bot, right_bot order
    apex_lists = [[True, True], [False, True], [True, False], [False, False]]
    check_bound_list = list()

    cropped_coord, idx, coord = np.array([lt, rt, lb, rb]), 0, np.array(coord)

    #how many accept box line error : config.crop_bound_threshold
    neg = config.neg_crop_bound_threshold
    pos = config.pos_crop_bound_threshold
    flags = [neg, pos]
    for apex_list in apex_lists:
        for flag in flags:
            apex = ((cropped_coord[idx] - coord[idx] <= flag) == apex_list).all()
            if apex: break

        check_bound_list.append(apex)
        idx += 1

    return check_bound_list

def coordinate_bounding_check(gt_list, gt_len, start_X, start_Y, width, height):
    new_gt_list = list()
    new_gt_len = 0
    left_top = np.array([start_X, start_Y])
    right_top = np.array([start_X + width, start_Y])
    left_bot = np.array([start_X, start_Y + height])
    right_bot = np.array([start_X + width, start_Y + height])

    for coord in gt_list:

        check_bound_list = adjust_cropped_coordinate(left_top, right_top, left_bot, right_bot, coord)

        if np.array(check_bound_list).all():
            # chk in box
            adjust_coord_x_y = np.array(
                [[start_X, start_Y], [start_X, start_Y], [start_X, start_Y], [start_X, start_Y]])
            coord = (np.array(coord) - adjust_coord_x_y).tolist()
            new_gt_list.append(coord)
            new_gt_len += 1

    return new_gt_list, new_gt_len


def crop(img, gt_list, gt_len):
    # default image size 768 x 768
    opt = random.randint(0, 4)
    height, width, _ = img.shape
    # BGR ->RGB
    img = img[:, :, ::-1]

    n_height, n_width = height // 2, width // 2
    cropped_img = img.copy()
    if opt == 0:  # left-top cropping

        cropped_img = cropped_img[:n_height, :n_width, :]
        new_gt, new_gt_len = coordinate_bounding_check(gt_list, gt_len, 0, 0, n_width, n_height)

    elif opt == 1:  # right-top cropping
        cropped_img = cropped_img[:n_height, n_width:, :]
        new_gt, new_gt_len = coordinate_bounding_check(gt_list, gt_len, n_width, 0, n_width, n_height)

    elif opt == 2:  # left-bottom cropping
        cropped_img = cropped_img[n_height:, :n_width, :]
        new_gt, new_gt_len = coordinate_bounding_check(gt_list, gt_len, 0, n_height, n_width, n_height)

    elif opt == 3:  # right-bottom cropping
        cropped_img = cropped_img[n_height:, n_width:, :]
        new_gt, new_gt_len = coordinate_bounding_check(gt_list, gt_len, n_width, n_height, n_width, n_height)

    elif opt == 4:  # center cropping
        half_n_height, half_n_width = n_height // 2, n_width // 2
        cropped_img = cropped_img[half_n_height:n_height + half_n_height, half_n_width:n_width + half_n_width, :]
        new_gt, new_gt_len = coordinate_bounding_check(gt_list, gt_len, half_n_width, half_n_height, n_width, n_height)
    new_gt_temp = new_gt[:]
    while True:
        if not new_gt_temp:
            break
        new_gt_element = new_gt_temp.pop()
        poly = np.array(new_gt_element).astype(np.int32).reshape((-1)).reshape(-1, 2)
        cv2.polylines(cropped_img, [poly.reshape((-1, 1, 2))], True, color=(255, 0, 0), thickness=2)
        ptColor = (0, 255, 255)

    cv2.imwrite('/home/hanish/workspace/debug_image/debug_crop.jpg', cropped_img)

    return cropped_img, new_gt, new_gt_len

#crop(img, arr_list, gt_len)
