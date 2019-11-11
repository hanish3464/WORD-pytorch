'''file_utils.py'''

# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2

def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path): 
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt' or ext =='.json':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    img_files.sort()
    mask_files.sort()
    gt_files.sort()
    return img_files, mask_files, gt_files

def saveResult(img_file, img, boxes, dir1='./prediction/', dir2='./gt/', verticals=None, texts=None):

        img = np.array(img)

        # make result file list
        filename, file_ext = os.path.splitext(os.path.basename(img_file))

        # result directory
        gt_file = dir2 + "res_" + filename + '.txt'
        predict_image_file = dir1 + "res_" + filename + '.jpg'

        with open(gt_file, 'w') as f:
            for i, box in enumerate(boxes):
                poly = np.array(box).astype(np.int32).reshape((-1))

                strResult = ','.join([str(p) for p in poly]) + '\r\n'
                f.write(strResult)
                poly = poly.reshape(-1, 2)
                cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(255, 0, 0), thickness=3)
                ptColor = (0, 255, 255)
                if verticals is not None:
                    if verticals[i]:
                        ptColor = (255, 0, 0)
                if texts is not None:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
                    cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)

        cv2.imwrite(predict_image_file, img)

def saveText(dir=None, text=None, index1=None):
    text_file = dir + index1 + '.txt'
    with open(text_file, 'w') as f:
        for i, box in enumerate(text):
            poly = np.array(box).astype(np.int32).reshape((-1))
            strResult = ','.join([str(p) for p in poly]) + '\r\n'
            f.write(strResult)

def adjustImageNum(idx, length):
    total_digits = len(str(length)) - 1
    cur_digits = len(str(idx)) - 1
    zeros = '0' * (total_digits - cur_digits)
    fixed_idx = zeros + str(idx)
    return fixed_idx

def saveImage(dir=None, img=None, index1=None, index2=None):
    if index2 is not None: cv2.imwrite(dir + index1 + '_' + index2 + '.png', img)
    else:cv2.imwrite(dir + index1 + '.png', img)

def saveMask(dir=None, heatmap=None, index1=None):
    heatmap = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imwrite(dir + index1 + '.png', heatmap)

def drawBBoxOnImage(dir=None, img=None, index1=None, boxes=None, flags=None):
    BBox_img = dir + index1 + '.png'
    for i, box in enumerate(boxes):
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        if flags == 'char': cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(255, 0, 0), thickness=1)
        if flags == 'word': cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=1)
        if flags == 'line': cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=1)
    cv2.imwrite(BBox_img, img)

