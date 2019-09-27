import numpy as np
import cv2


def adjust_resized_coordinate(gt, gt_len, Rx, Ry):
    gt = np.array(gt).reshape(-1,4,2).astype(np.float32)
    #print(gt)
    #print(Rx,Ry)
    for i in range(gt_len):
        gt[i][:, 0] *= Rx
        gt[i][:, 1] *= Ry
    #print(gt)
    return gt.astype(int).tolist()

def coord_min_and_max(gt, gt_len):
    gt_list = list()
    for i in range(gt_len):
        gt_list.append(np.append(gt[i][0], gt[i][2]).reshape(-1, 4).tolist())
    gt_list = np.array(gt_list).reshape(-1, 4).tolist()
    return gt_list

def resize_gt(img, gt, gt_len):
    resized_img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    gt_list = list()
    H, W, C = img.shape
    nH, nW = 512, 512
    Rx = float(nW) / W
    Ry = float(nH) / H

    for idx in range(gt_len):
        if len(gt[0]) == 4:
            gt = np.array(gt).reshape(-1, 2)
            gt_4_coord = np.array(
                [gt[0][0], gt[0][1], gt[1][0], gt[0][1], gt[1][0], gt[1][1], gt[0][0], gt[1][1]]).reshape(-1, 4, 2)
        else:
            gt_4_coord = gt[0]
        gt_list.append(gt_4_coord)
        gt = gt.reshape(-1, 4)
        gt = gt[1:]
    resized_gt_list = adjust_resized_coordinate(gt_list, gt_len, Rx, Ry)

    return resized_img, resized_gt_list

def resize_psd(img, gt, gt_len, idx, name):
    resized_img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite('./psd/resized_jpg_images/' + name + '.jpg', resized_img)
    gt_list = list()
    H, W, C = img.shape
    nH, nW = 512, 512
    #Rx = float(nH) / H
    #Ry = float(nW) / W
    Rx = float(nW) / W
    Ry = float(nH) / H

    #print(gt,gt_len)
    for idx in range(gt_len):
        if len(gt[0]) == 4:
            gt = np.array(gt).reshape(-1, 2)
            gt_4_coord = np.array([gt[0][0],gt[0][1],gt[1][0],gt[0][1],gt[1][0],gt[1][1],gt[0][0],gt[1][1]]).reshape(-1,4,2)
        else: gt_4_coord = gt[0]
        gt_list.append(gt_4_coord)
        gt = gt.reshape(-1,4)
        gt = gt[1:]
    resized_gt_list = adjust_resized_coordinate(gt_list, gt_len, Rx, Ry)

    return resized_img, resized_gt_list
