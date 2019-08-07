""" evaluation.py """
"""   
After comparing model prediction(pdt) with ground_truth(gt),
It will be calculated as IoU, Precision, Recall, and F1_Score
"""

import codecs
import numpy as np
import sys
import cv2

import config

confusion_matrix = {'true_pos': 0, 'true_neg': 0, 'false_pos': 0, 'false_neg': 0}

def calculate_precision():
    if (confusion_matrix['true_pos'] + confusion_matrix['false_pos']) > 0:
        precision = float(confusion_matrix['true_pos']) / (confusion_matrix['true_pos'] + confusion_matrix['false_pos'])
    else:
        precision = 0

    return precision

def calculate_recall():
    if(confusion_matrix['true_pos'] + confusion_matrix['false_neg']) > 0:
        recall = float(confusion_matrix['true_pos']) / (confusion_matrix['true_pos'] + confusion_matrix['false_neg'])
    else:
        recall = 0

    return recall

def load_box_max_coordinate(path, idx):

    boxes = list() ; max_boxes=list() ; len = 0
    with codecs.open(path + str(idx) + '.txt', encoding='utf-8_sig') as file:
        coordinate = file.readlines()
        #print('json_txt_label')
        for line in coordinate:
            len += 1
            tmp = line.split(',')
            arr_coordinate = [int(n) for n in tmp]

            box = np.array(arr_coordinate).reshape([1,4,2])
            box_max = box.max(axis=1)

            boxes = np.append(boxes, box).astype(int)
            max_boxes = np.append(max_boxes, box_max).astype(int)

    return max_boxes, boxes, len

def pop_coordinate_in_boxes(boxes, max_boxes):

    width, height, max_boxes = max_boxes[0], max_boxes[1], max_boxes[2:]
    tmp_canvas = np.zeros((height, width))
    canvas = tmp_canvas.copy()
    box, boxes = boxes[0:8], boxes[8:]
    box = np.array(box).reshape([1, 4, 2])
    return width, height, box, boxes, max_boxes, canvas

def calculate_intersection_and_union(width_max, height_max, gt_box, pdt_box, gt_box_area, pdt_box_area):

    canvas = np.zeros((height_max + 1, width_max + 1))
    cv2.drawContours(canvas, gt_box, -1, 1, thickness=-1)
    cv2.drawContours(canvas, pdt_box, -1, 1, thickness=-1)
    union = np.sum(canvas) - 2
    intersection = gt_box_area + pdt_box_area - union
    return intersection / union

def IOU(gt_path, pdt_path, idx, status):

    gt_max_boxes, gt_boxes, gt_len = load_box_max_coordinate(gt_path, idx)
    pdt_max_boxes, pdt_boxes, pdt_len = load_box_max_coordinate(pdt_path, idx)

    tmp_gt_max_boxes = gt_max_boxes
    tmp_gt_boxes = gt_boxes

    for i in range(pdt_len):
        flag = False

        pdt_w, pdt_h, pdt_box, pdt_boxes, pdt_max_boxes, pdt_canvas = pop_coordinate_in_boxes(pdt_boxes, pdt_max_boxes)
        pdt_box_area = np.sum(cv2.drawContours(pdt_canvas, pdt_box, -1, 1, thickness=-1))

        for j in range(gt_len):
            gt_w, gt_h, gt_box, gt_boxes, gt_max_boxes, gt_canvas = pop_coordinate_in_boxes(gt_boxes, gt_max_boxes)
            gt_box_area = np.sum(cv2.drawContours(gt_canvas, gt_box, -1, 1, thickness=-1))

            w_max = max(gt_w, pdt_w)
            h_max = max(gt_h, pdt_h)

            iou = calculate_intersection_and_union(w_max, h_max, gt_box, pdt_box, gt_box_area, pdt_box_area)

            if iou > config.iou_threshold:
                if status == 'Precision':
                    confusion_matrix['true_pos'] += 1
                flag = True
                break

        if not flag:
            if status == 'Precision':
                confusion_matrix['false_pos'] += 1
            if status == 'Recall':
                confusion_matrix['false_neg'] += 1

        gt_max_boxes = tmp_gt_max_boxes
        gt_boxes = tmp_gt_boxes

def evaluation():
    """MODEL EVALUATION"""

    for idx in range(1, config.img_num):
        IOU(config.json_gt_folder, config.ground_truth_folder, idx, 'Precision')
        IOU(config.ground_truth_folder, config.json_gt_folder, idx, 'Recall')

        precision = round(calculate_precision(), 6)
        recall = round(calculate_recall(), 6)
        print('[TEST IMG : ({}/{})] Precision: {} recall : {}'.format(idx, config.img_num-1, precision, recall))


