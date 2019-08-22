import cv2
import numpy as np
import preprocess
import config
import gaussian
import sys

np.set_printoptions(threshold=sys.maxsize)
temp_img = config.train_images_folder_path + 'res_1.jpg'
img = preprocess.loadImage(temp_img)
temp_gt = config.train_ground_truth_folder + '1.jpg.txt'
gt, gt_len = preprocess.loadText(temp_gt)


def gaussian():  # some bugs exist ; it can generate borderline by adjusting spread value

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

    cv2.imwrite('./gauss/gauss_img.jpg', isotropicGaussian2dMap)
    cv2.imwrite('./gauss/adjust_img.jpg', adjust_gaussian_heat_map)
    return adjust_gaussian_heat_map


def perspective_transform(image, gt):

    max_x, max_y = np.max(gt[:, 0]).astype(np.int32), np.max(gt[:, 1]).astype(np.int32)
    dst = np.array([
        [0, 0], [image.shape[1] - 1, 0], [image.shape[1] - 1, image.shape[0] - 1], [0, image.shape[0] - 1]],
        dtype="float32")
    print(gt)
    #print(dst)
    #print(max_x,max_y)
    M = cv2.getPerspectiveTransform(dst, gt)
    print(M)
    warped = cv2.warpPerspective(image, M, (max_x, max_y))
    #print(warped.shape)
    cv2.imwrite('./gauss/warp_img.jpg', warped)
    return warped


def add_character(image, bbox, gaussian_heat_map):
    bbox = np.array(bbox)
    if np.any(bbox < 0) or np.any(bbox[:, 0] > image.shape[1]) or np.any(bbox[:, 1] > image.shape[0]):
        return image

    top_left = np.array([np.min(bbox[:, 0]), np.min(bbox[:, 1])]).astype(np.int32)
    bbox -= top_left[None, :]
    transformed = perspective_transform(gaussian_heat_map.copy(), bbox.astype(np.float32))

    start_row = max(top_left[1], 0) - top_left[1]
    start_col = max(top_left[0], 0) - top_left[0]
    end_row = min(top_left[1] + transformed.shape[0], image.shape[0])  # H
    end_col = min(top_left[0] + transformed.shape[1], image.shape[1])  # W

    image[max(top_left[1], 0):end_row, max(top_left[0], 0):end_col] += transformed[start_row:end_row - top_left[1],
                                                                       start_col:end_col - top_left[0]]
    return image


def generate_text_region(img, gt, gt_len):
    gaussian_heat_map = gaussian()
    h, w, _ = img.shape
    target = np.zeros([h, w], dtype=np.float32)

    # generate text_region_GT
    for i in range(gt_len):
        target = add_character(target, gt[i], gaussian_heat_map).astype(float)
        target_tmp = cv2.applyColorMap(target.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite('./gauss/region_' + str(i) + '.jpg', target_tmp)
    cv2.imwrite('./gauss/final_region_img.jpg', target_tmp)


def add_affinity(image, gt, gt_next, gaussian_heat_map):
    center_gt, center_gt_next = np.mean(gt, axis=0), np.mean(gt_next, axis=0)
    top_triangle_center_gt = np.mean([gt[0], gt[1], center_gt], axis=0)
    bot_triangle_center_gt = np.mean([gt[2], gt[3], center_gt], axis=0)
    top_triangle_center_gt_next = np.mean([gt_next[0], gt_next[1], center_gt_next], axis=0)
    bot_triangle_center_gt_next = np.mean([gt_next[2], gt_next[3], center_gt_next], axis=0)

    affinity_box = np.array(
        [top_triangle_center_gt, bot_triangle_center_gt, top_triangle_center_gt_next, bot_triangle_center_gt_next])
    #print(affinity_box)
    return add_character(image, affinity_box, gaussian_heat_map)

def generate_affinity_region(img, gt, gt_len):
    gaussian_heat_map = gaussian()
    h, w, _ = img.shape
    print(h,w)
    target = np.zeros([h, w], dtype=np.float32)

    # generate affinity_region_GT
    for i in range(gt_len-1):
        target = add_affinity(target, gt[i], gt[i+1], gaussian_heat_map).astype(float)
        target_tmp = cv2.applyColorMap(target.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite('./gauss/affinity_' + str(i) + '.jpg', target_tmp)
    cv2.imwrite('./gauss/final_affinity_img.jpg', target_tmp)

#generate_text_region(img, gt, gt_len)
generate_affinity_region(img, gt, gt_len)
