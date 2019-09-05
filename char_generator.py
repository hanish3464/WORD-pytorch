'''char_generator.py'''

#-*- coding: utf-8 -*-

import cv2
import numpy as np 
import preprocess
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from collections import OrderedDict
import config
import file
from wtd import WTD
import os
import debug
import math
import time

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def word_patches_cropper(img, gt, gt_len, img_path):
    cropped_img_list = list()
    for idx in range(gt_len):
        x_min, y_min, x_max, y_max = gt[idx]
        cropped_img = img[y_min:y_max, x_min:x_max]
        filename, file_ext = os.path.splitext(os.path.basename(img_path))
        cropped_file = config.jpg_cropped_images_folder_path + filename + '_' + str(idx) + '.jpg'
        cv2.imwrite(cropped_file, cropped_img)
        cropped_img_list.append(cropped_img)
    print(np.array(cropped_img_list).shape)

    return cropped_img

def char_postprocess(text_map, affinity_map, img):
    # prepare data
    print(text_map.shape)
    affinity_map = affinity_map.copy()  # link score heatamp
    text_map = text_map.copy()  # text score heatmap
    img_h, img_w = text_map.shape

    """ labeling method """
    ret, text_score = cv2.threshold(text_map, config.low_text, 1, 0)
    # debug.printing(text_score)
    ret, link_score = cv2.threshold(affinity_map, config.link_threshold, 1, 0)
    # debug.printing(link_score)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    # debug.printing(text_score_comb)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8),
                                                                         connectivity=4)
    # debug.printing(labels)

    # nLabels = the number of labels
    # label map has number per segmentation unit which is number order
    det = []
    mapper = []
    """Debug Global Map"""
    glbMap = np.zeros(text_map.shape, dtype=np.uint8)
    for k in range(1, nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue

        # thresholding
        if np.max(text_map[labels == k]) < config.text_threshold: continue

        # make segmentation map
        segmap = np.zeros(text_map.shape, dtype=np.uint8)  # empty map create

        segmap[labels == k] = 255  # if label(segmentation unit) number matches k number, it checks 255
        # debug.printing(segmap)
        segmap[np.logical_and(link_score == 1, text_score == 0)] = 0  # remove link area
        #debug.printing(segmap)
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        print(x,y,w,h)
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # both margin of bounding box is initialized

        # boundary check
        if sx < 0: sx = 0  # if image pixel extends left, it sets 0
        if sy < 0: sy = 0  # if image pixel extends bottom, it sets 0
        if ex >= img_w: ex = img_w  # if image pixel extends width, it sets width
        if ey >= img_h: ey = img_h  # if image pixel extends height, it sets height

        # niter size is RECTANGLE shape kernel(filter)
        # maybe kernel is part of bounding box in section.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        # segmentation blobs are covered with kernel size(rectangle filter)
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)
        # debug.printing(segmap)
        glbMap += segmap
        # make box
        np_contours = np.roll(np.array(np.where(segmap != 0)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)
        bl, tl, tr, br = box
        box = np.array([tl,tr,br,bl]).astype(int)
        img = cv2.resize(img, (171,112), interpolation=cv2.INTER_LINEAR)
        print(box)
        print(box.shape)
        box = box.reshape(-1, 2)
        cv2.polylines(img, [box.reshape((-1, 1, 2))], True, color=(255, 0, 0), thickness=1)
        cv2.imwrite('./psd/psd_draw_box_img.jpg', img)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)
        det.append(box)
        mapper.append(k)
    # box creation
    debug.printing(glbMap)
    #return det, labels, mapper

def gen_character(generator, cropped_img):

    #prerprocess
    x = preprocess.normalizeMeanVariance(cropped_img)
    x = torch.from_numpy(x).permute(2,0,1)
    x = Variable(x.unsqueeze(0))

    '''GPU'''
    if config.cuda: x= x.cuda()

    '''predict character with pretrained model'''
    y, _ = generator(x)
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_affinity = y[0, :, :, 0].cpu().data.numpy()
    cv2.imwrite('./psd/score_text.jpg', score_text)

    '''postprocess'''
    char_postprocess(score_text, score_affinity, cropped_img)

    #return bboxes, polys, score_text

def pseudo_gt_generator():
    generator = WTD()
    print('We use pretrained model to divide cha annotation from PSD')
    print('Pretrained Path : ' + config.pretrained_model_path)
    if config.cuda: #GPU
        generator.load_state_dict(copyStateDict(torch.load(config.pretrained_model_path)))
    else:  # ONLY CPU
        generator.load_state_dict(copyStateDict(torch.load(config.pretrained_model_path, map_location='cpu')))

    if config.cuda:
        generator = generator.cuda()
        generator = torch.nn.DataParallel(generator)
        cudnn.benchmark = False

    generator.eval()

    #jpg folder path divided from PSD file
    image_list, _, _ = file.get_files(config.jpg_images_folder_path)
    _, _ , gt_list = file.get_files(config.jpg_text_ground_truth)

    if not os.path.isdir(config.jpg_cropped_images_folder_path):
        os.mkdir(config.jpg_cropped_images_folder_path)
    if not os.path.isdir(config.jpg_text_ground_truth):
        os.mkdir(config.jpg_text_ground_truth)

    datasets = {'images': image_list, 'gt': gt_list}

    for idx in range(len(datasets['images'])):

        img_list = preprocess.loadImage(datasets['images'][idx])
        gt_list, length = preprocess.loadText(datasets['gt'][idx])
        cropped_img = word_patches_cropper(img_list, gt_list, length, datasets['images'][idx])
        gen_character(generator, cropped_img)
