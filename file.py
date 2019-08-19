# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import preprocess
import debug
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
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        img = np.array(img)

        # make result file list
        filename, file_ext = os.path.splitext(os.path.basename(img_file))

        # result directory
        gt_file = dir2 + "res_" + filename + '.txt'
        predict_image_file = dir1 + "res_" + filename + '.jpg'

        #gt_txt file generation:
        with open(gt_file, 'w') as f:
            for i, box in enumerate(boxes):
                poly = np.array(box).astype(np.int32).reshape((-1))
                #debug.getScalaValue(poly)
                strResult = ','.join([str(p) for p in poly]) + '\r\n'
                #debug.getScalaValue(strResult)
                f.write(strResult)
                #drawing bounding ploy type box with CV
                poly = poly.reshape(-1, 2)
                cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(255, 0, 0), thickness=3)
                ptColor = (0, 255, 255)
                if verticals is not None: #this condition is not executed
                    if verticals[i]:
                        ptColor = (255, 0, 0)
                # drawing bounding normal rect box with CV
                if texts is not None: #this condition is not executed
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
                    cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)


        # Save result image
        cv2.imwrite(predict_image_file, img)
        #debug.printing(img)


