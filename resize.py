import numpy as np
import cv2


def adjust_resized_coordinate(gt, gt_len, Rx, Ry):
    gt = np.array(gt).astype(float)
    for i in range(gt_len):
        gt[i][:, 0] *= Rx
        gt[i][:, 1] *= Ry
    return gt.astype(int).tolist()


def resize(img, gt, gt_len):
    resized_img = cv2.resize(img, (512, 512))
    H, W, C = img.shape
    nH, nW = 512, 512
    Rx = float(nH) / H
    Ry = float(nW) / W

    print(gt)
    if len(gt[0]) == 4:
        gt = np.array(gt).reshape(-1,2)
        gt = np.array([gt[0][0],gt[0][1],gt[1][0],gt[0][1],gt[1][0],gt[1][1],gt[0][0],gt[1][1]]).reshape(-1,4,2)
    print(gt)
    resized_gt = adjust_resized_coordinate(gt, gt_len, Rx, Ry)

    return resized_img, resized_gt
