import cv2
import random
import config
import preprocess
import numpy as np

temp_img = config.test_images_folder_path + '0001-001.jpg'
img = preprocess.loadImage(temp_img)
temp_gt = config.test_ground_truth + '1.txt'
arr_list, gt_len = preprocess.loadText(temp_gt)
print(arr_list)
def coordinate_bounding_check(gt_list, gt_len, start_X, start_Y, width, height):
    left_top = np.array([start_X, start_Y])
    right_top = np.array([start_X + width, start_Y])
    left_bot = np.array(start_X, start_Y + height])
    right_bot = np.array([start_X + width, start_Y + height])
    print(left_top, right_top, left_bot, right_bot)

    for i in range(gt_len):
        for j in gt_list[i]:
            lt = ((left_top - np.array(j[0]) <= 0).all()
            rt = right_top - np.array(j[1])
            rb = right_bot - np.array(j[2])
            lb = left_bot - np.array(j[3])

            if lt & rt & rb & lb is True:
                #chk in box
            else:
                #pop in box

def crop(img,  gt_list, gt_len):
    #default image size 768 x 768
#    opt = random.randint(0,4)
    opt = 0
    height, width, _ = img.shape
    #BGR ->RGB
    img = img[:,:,::-1]

    n_height, n_width = height//2, width//2

    if opt == 0: #left-top cropping
        cropped_img = img.copy()
        cropped_img = cropped_img[:n_height, :n_width, :]
        coordinate_bounding_check(gt_list, gt_len, 0, 0, n_width, n_height) 

    elif opt == 1: #right-top cropping
        cropped_img = img.copy()
        cropped_img = cropped_img[:n_height, n_width:, :]

    elif opt == 2: #left-bottom cropping
        cropped_img = img.copy()
        cropped_img = cropped_img[n_height:, :n_width, :]

    elif opt == 3: #right-bottom cropping
        cropped_img = img.copy()
        cropped_img = cropped_img[n_height:, n_width:, :]

    elif opt == 4: #center cropping
        half_n_height, half_n_width = n_height//2, n_width//2
        cropped_img = img.copy()
        cropped_img = cropped_img[half_n_height:n_height+half_n_height, half_n_width:
                n_width+half_n_width, :]

    small_img = cv2.resize(img, (260,800), interpolation =cv2.INTER_LINEAR)
    small_crop_img = cv2.resize(cropped_img, (260,800), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('origin', small_img)
    cv2.imshow('crop', small_crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
    
crop(img, arr_list, gt_len)
