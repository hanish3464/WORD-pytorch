from PIL import Image
import numpy as np
import config
import random
import cv2

def rotate_img(img, angle):
    h, w = img.shape[:2]
    center_x, center_y = w//2 , h//2
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    new_W = int((h * sin) + (w * cos))
    new_H = int((h * cos) + (w * sin))

    M[0, 2] += (new_W / 2) - center_x
    M[1, 2] += (new_H / 2) - center_y

    return cv2.warpAffine(img, M, (new_W, new_H)), h, w, center_x, center_y

def rotate_gt(gt, cx, cy, h, w, angle):
    new_gt = list(gt)
    for i, coord in enumerate(gt):

        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 0])

        nW = float((h * sin) + (w * cos))
        nH = float((h * cos) + (w * sin))

        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy

        v = [coord[0], coord[1], 1]
        calculated = np.dot(M, v)
        new_gt[i] = [calculated[0], calculated[1]]

    return new_gt


def rotate(img, gt, gt_len):
    angle = random.randint(0, 11) * 30
    rotated_img, height, width, center_x, center_y = rotate_img(img, angle)
    rotated_gt = list()
    for idx in range(gt_len):
            rotated_gt.append(rotate_gt(gt[idx], center_x, center_y, height, width, angle))

    return rotated_img, rotated_gt


