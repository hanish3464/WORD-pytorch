import cv2
import numpy as np
import math
import preprocess
import config
import gaussian
import sys
import os
import debug

# np.set_printoptions(threshold=sys.maxsize)
# temp_img = config.train_images_folder_path + 'res_1.jpg'
# img = preprocess.loadImage(temp_img)
# temp_gt = config.train_ground_truth_folder + '1.jpg.txt'
# gt, gt_len = preprocess.loadText(temp_gt)

def gaussian():  # some bugs exist ; it may generate borderline by adjusting spread value

    sigma = config.gaussian_sigma
    spread = config.gaussian_spread

    extent = int(spread * sigma)
    isotropicGaussian2dMap = np.zeros((2 * extent, 2 * extent), dtype=np.float32)

    for i in range(2 * extent):
        for j in range(2 * extent):
            isotropicGaussian2dMap[i, j] = float(1) / 2 / np.pi / (sigma ** 2) * np.exp(
                -1 / 2 * ((i - spread * sigma - 0.5) ** 2 + (j - spread * sigma - 0.5) ** 2) / (sigma ** 2))
    isotropicGaussian2dMap = (isotropicGaussian2dMap / np.max(isotropicGaussian2dMap) * 255).astype(np.uint8)

    # repair cropped pixel value
    h, w = isotropicGaussian2dMap.shape

    adjust_gaussian_heat_map = np.zeros((h + 2, w + 2)).astype(np.uint8)
    adjust_gaussian_heat_map[:h, :w] = isotropicGaussian2dMap[:, :]
    adjust_gaussian_heat_map[:h, w] = isotropicGaussian2dMap[:, 1]
    adjust_gaussian_heat_map[:h, w + 1] = isotropicGaussian2dMap[:, 0]
    adjust_gaussian_heat_map[h + 1] = adjust_gaussian_heat_map[0]
    adjust_gaussian_heat_map[h] = adjust_gaussian_heat_map[1]


    #cv2.imwrite('./gauss/gauss_img.jpg', isotropicGaussian2dMap)
    #cv2.imwrite('./gauss/adjust_img.jpg', adjust_gaussian_heat_map)
    return adjust_gaussian_heat_map


def perspective_transform(gauss, cha_box, i, flags = None):
    #image :
    #help_perspective.four_point_transform(gauss, cha_box)
    #flags = 'text'
    print(cha_box)
    if flags == 'text':
        #max_x, max_y = np.max(cha_box[:, 0]).astype(np.int32), np.max(cha_box[:, 1]).astype(np.int32)
        max_x, max_y = np.int32(math.ceil(np.max(cha_box[:, 0]))), np.int32(math.ceil(np.max(cha_box[:, 1])))
    if flags == 'affinity':
        #max_x, max_y= np.max(cha_box[:, 0]).astype(np.int32), np.max(cha_box[:, 1]).astype(np.int32)
        max_x, max_y = np.int32(math.ceil(np.max(cha_box[:, 0]))), np.int32(math.ceil(np.max(cha_box[:, 1])))
        print(cha_box)
        print(max_x, max_y)
        x_center1, y_center1 = sum(cha_box[:2,0])/float(2), sum(cha_box[:2,1])/float(2)
        x_center2, y_center2 = sum(cha_box[2:4,0])/float(2), sum(cha_box[2:4,1])/float(2)
        affinity_tl = (cha_box[0, 0] + x_center1) / float(2), (cha_box[0, 1] + y_center1) / float(2)
        affinity_tr = (cha_box[1, 0] + x_center1) / float(2), (cha_box[1, 1] + y_center1) / float(2)
        affinity_br = (cha_box[2, 0] + x_center2) / float(2), (cha_box[2, 1] + y_center2) / float(2)
        affinity_bl = (cha_box[3, 0] + x_center2) / float(2), (cha_box[3, 1] + y_center2) / float(2)
        #print(affinity_tl,affinity_tr,affinity_br, affinity_bl)
        tr = np.array(affinity_tr) - np.array(affinity_tl)
        br = np.array(affinity_br) - np.array(affinity_tl)
        bl = np.array(affinity_bl) - np.array(affinity_tl)
        new_aff = np.array([[0,0], tr, br, bl], np.float32)

        cha_box = new_aff[:]
        # print(new_aff)
        # print(cha_box)
        # print(cha_box.shape)
        # print(new_aff.shape)
    h, w = gauss.shape[:2]
    #gaussian region size -> character_box size
    gauss_region = np.array([[0, 0], [w - 1, 0], [h - 1, w - 1], [0, h - 1]], dtype="float32")
    #print(gauss_region)
    M = cv2.getPerspectiveTransform(src= gauss_region, dst = cha_box)
    warped = cv2.warpPerspective(gauss,  M, (max_x, max_y), borderValue = 0, borderMode=cv2.BORDER_CONSTANT)
    #cv2.imwrite('./gauss/warp_img' + str(i) + '.jpg', warped)
    return warped


def add_character(image, bbox, gaussian_heat_map, i, flags = None):
    bbox = np.array(bbox)
    if np.any(bbox < 0) or np.any(bbox[:, 0] > image.shape[1]) or np.any(bbox[:, 1] > image.shape[0]):
        return image

    top_left = np.array([np.min(bbox[:, 0]), np.min(bbox[:, 1])]).astype(np.int32)
    bbox -= top_left[None, :]
    transformed = perspective_transform(gaussian_heat_map.copy(), bbox.astype(np.float32), i, flags)

    start_row = max(top_left[1], 0) - top_left[1]
    start_col = max(top_left[0], 0) - top_left[0]
    end_row = min(top_left[1] + transformed.shape[0], image.shape[0])  # H
    end_col = min(top_left[0] + transformed.shape[1], image.shape[1])  # W

    image[max(top_left[1], 0):end_row, max(top_left[0], 0):end_col] += transformed[start_row:end_row - top_left[1],
                                                                       start_col:end_col - top_left[0]]
    return image

def text_box_valid_check(box, path):

    filename, file_ext = os.path.splitext(os.path.basename(path))
    word_gt_path = './psd/word_ground_truth/' + filename + file_ext
    word_gt, word_gt_len = preprocess.loadText(word_gt_path)
    apex_lists = [[True, True], [False, True], [False, False], [True, False]]
    word_gt, box, idx = np.array(word_gt), np.array(box), 0
    check_bound_list = list()
    neg = config.neg_link_threshold
    pos = config.pos_link_threshold
    flags = [neg, pos]
    for i in range(word_gt_len):
        for apex_list in apex_lists:
            for flag in flags:
                apex = ((word_gt[i][idx] - box[idx] <= flag) == apex_list).all()
                if ((word_gt[i][idx] - box[idx] == 0) == [True,True]).all(): # word boundary == char boundary
                    apex = True
                if apex:
                    break
            check_bound_list.append(apex)
            idx +=1
        if np.array(check_bound_list).all():
            return np.append(word_gt[i], box).tolist()
        check_bound_list[:] = list()
        idx = 0

    return False


def generate_text_region(img, gt, gt_len, k, path):
    gaussian_heat_map = gaussian()
    h, w, _ = img.shape
    target = np.zeros([h, w], dtype=np.float32)
    #gt, gt_len = preprocess.sort_gt(gt, gt_len)
    # generate text_region_GT
    box_in_word = list()
    for i in range(gt_len):
        box_in_word.append(text_box_valid_check(gt[i], path))
        #target = add_character(target, gt[i], gaussian_heat_map, i, flags='text').astype(float)
        #target_tmp = cv2.applyColorMap(target.astype(np.uint8), cv2.COLORMAP_JET)
        #cv2.imwrite('./gauss/region_' + str(i) + '.jpg', target_tmp)
    #cv2.imwrite('./gauss/final_region_img' + str(k) + '.jpg', target_tmp)
    #print(len(box_in_word))
    preprocess.sort_gt(box_in_word, len(box_in_word))
def affinity_box_valid_check(box, path):
    #print(box)
    filename, file_ext = os.path.splitext(os.path.basename(path))
    word_gt_path = './psd/word_ground_truth/' + filename + file_ext
    word_gt, word_gt_len = preprocess.loadText(word_gt_path)
    #print(word_gt)
    apex_lists = [[True,True], [False,True], [False, False], [True,False]]
    word_gt, box, idx = np.array(word_gt), np.array(box), 0
    check_bound_list = list()
    neg = config.neg_link_threshold
    pos = config.pos_link_threshold
    flags =[neg, pos]
    for i in range(word_gt_len):
        for apex_list in apex_lists:
            for flag in flags:
                #print(word_gt[i][idx], box[idx])
                apex = ((word_gt[i][idx] - box[idx] <= flag) == apex_list).all()

                if apex:
                    #print("apexTrue")
                    break
            check_bound_list.append(apex)
            #print(check_bound_list)
            idx +=1
        if np.array(check_bound_list).all(): return True
        check_bound_list[:] = list()
        idx = 0
    else: return False

def add_affinity(image, gt, gt_next, gaussian_heat_map, i, path):

    center_gt, center_gt_next = np.mean(gt, axis=0), np.mean(gt_next, axis=0)
    top_triangle_center_gt = np.mean([gt[0], gt[1], center_gt], axis=0)
    bot_triangle_center_gt = np.mean([gt[2], gt[3], center_gt], axis=0)
    top_triangle_center_gt_next = np.mean([gt_next[0], gt_next[1], center_gt_next], axis=0)
    bot_triangle_center_gt_next = np.mean([gt_next[2], gt_next[3], center_gt_next], axis=0)

    affinity_box = np.array(
        [top_triangle_center_gt, top_triangle_center_gt_next, bot_triangle_center_gt_next, bot_triangle_center_gt])
    print(affinity_box)
    if affinity_box_valid_check(affinity_box, path):
        print("--------------------------True-----------------------------------------")
        return add_character(image, affinity_box, gaussian_heat_map, i, flags = 'affinity')
    else:
        print('--------------------------False----------------------------------------')
        return image

def generate_affinity_region(img, gt, gt_len, k, path):
    gaussian_heat_map = gaussian()
    h, w, _ = img.shape
    #print(h,w)
    target = np.zeros([h, w], dtype=np.float32)

    # generate affinity_region_GT
    for i in range(gt_len-1):
        target = add_affinity(target, gt[i], gt[i+1], gaussian_heat_map, i, path).astype(float)
        target_tmp = cv2.applyColorMap(target.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite('./gauss/affinity_' + str(i) + '.jpg', target_tmp)
    cv2.imwrite('./gauss/final_affinity_img' + str(k) +'.jpg', target_tmp)

for i in range(1,8):
    temp_img = './psd/resized_jpg_images/' + 'tmp' + str(i) + '.jpg'
    img = preprocess.loadImage(temp_img)
    temp_gt = './psd/char_ground_truth/' + 'tmp' + str(i) + '.txt'
    gt, gt_len = preprocess.loadText(temp_gt)
    generate_text_region(img, gt, gt_len, i, temp_gt)
    #generate_affinity_region(img, gt, gt_len, i, temp_gt)
