'''file_utils.py'''

# -*- coding: utf-8 -*-
import os
import opt
import numpy as np
import cv2
import codecs
import json
import shutil

def get_files(img_dir):
    imgs, masks, xmls, names = list_files(img_dir)
    return imgs, masks, xmls, names


def list_files(in_path): 
    img_files = []
    mask_files = []
    gt_files = []
    name_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            name_files.append(filename)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt' or ext =='.json':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    name_files.sort()
    img_files.sort()
    mask_files.sort()
    gt_files.sort()
    return img_files, mask_files, gt_files, name_files


def rm_all_dir(dir=None):
    if os.path.isdir(dir): shutil.rmtree(dir)

def mkdir(dir=None):
    for folder in dir:
        if not os.path.isdir(folder):
            os.mkdir(folder)


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


def saveText(save_to=None, text=None, name=None):
    text_file = save_to + name + '.txt'
    with open(text_file, 'w') as f:
        for i, box in enumerate(text):
            if box == []: continue
            poly = np.array(box).astype(np.int32).reshape((-1))
            strResult = ','.join([str(p) for p in poly]) + '\r\n'
            f.write(strResult)


def resultNameNumbering(origin=None, digit=None):
    total_digits = len(str(digit)) - 1
    cur_digits = len(str(origin)) - 1
    zeros = '0' * (total_digits - cur_digits)
    fixed_idx = zeros + str(origin)
    return fixed_idx


def saveImage(save_to=None, img=None, index1=None, index2=None, ext=None):
    if type(index1) == int: index1 = str(index1)
    if type(index2) == int: index2 = str(index2)

    if index2 is not None: cv2.imwrite(save_to + index1 + '_' + index2 + ext, img)
    else: cv2.imwrite(save_to + index1 + ext, img)


def saveAllImages(save_to=None, imgs=None, index1=None, ext=None):
    if type(index1) == int: index1 = str(index1)
    if index1 is not None:
        for idx2, img in enumerate(imgs):
            cv2.imwrite(save_to + index1 + '_' + str(idx2) + ext, img)
    else:
        for idx1, img in enumerate(imgs):
            cv2.imwrite(save_to + str(idx1) + ext, img)


def saveMask(dir=None, heatmap=None, index1=None):
    heatmap = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imwrite(dir + index1 + '.png', heatmap)


def drawBBoxOnImage(dir=None, img=None, index1=None, boxes=None, flags=None):
    BBox_img = dir + index1 + '.png'
    for i, box in enumerate(boxes):
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        if flags == 'char': cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2)
        if flags == 'word': cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=1)
        if flags == 'line': cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(255, 0, 0), thickness=1)
        cv2.imwrite(BBox_img, img)


def loadJson(json_file):
    items = []
    with open(json_file) as data_file:
        data = json.load(data_file)
        for idx, info in enumerate(data['shapes']):
            labels = info['label']; bboxes = info['points']
            items.append([labels, bboxes])
    return items


def saveCSV(dir=None, dst=None, index=None, label=None, num=None):
    distorted_image_file = dir + str(index) + '.png'
    dst.write(u'{},{},{}\n'.format(distorted_image_file, label, num))


def createCustomCSVFile(src=None, files=None, gt=None, nums=None):
    labels_csv = codecs.open(os.path.join(src), 'w', encoding='utf-8')
    for k, file in enumerate(files):
        labels_csv.write(u'{},{},{}\n'.format(file, gt[k-1], nums[k-1]))


def loadText(txt_file):
    arr = []
    with codecs.open(txt_file, encoding='utf-8_sig') as file:
        lines= file.readlines()
        for line in lines: arr.append(line.strip('\r\n'))
    return arr


def loadSpacingWordInfo(load_from=None):
    arr_list = list()
    length = 0
    with codecs.open(load_from, encoding='utf-8_sig') as file:
        coordinate = file.readlines()
        for line in coordinate:
            tmp = line.split(',')
            tmp[-1] = tmp[-1].strip('\r\n')
            if tmp[0] == '': continue
            arr_coordinate = [int(n) for n in tmp]
            coordinate = np.array(arr_coordinate).tolist()
            arr_list.append(coordinate)
            length += 1
    return arr_list, length


def makeLabelMapper(load_from=None):
    label_map = loadText(load_from)
    label_num = np.arange(len(label_map))
    label_mapper = np.vstack((np.array(label_map), label_num))
    return label_mapper


