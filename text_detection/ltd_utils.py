"""
ltd_utils.py
"""
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math
import opt


def link_refine(boxes=None, MARGIN=None):
    reshape_boxes = []
    s = 0
    for i in range(len(boxes)):
        if i + 1 == len(boxes):
            reshape_boxes.append(boxes[s:i + 1])
            break
        if abs(boxes[i][-1] - boxes[i + 1][-1]) >= 15:
            reshape_boxes.append(boxes[s:i + 1])
            s = i + 1
    line_boxes = []
    for reshape_box in reshape_boxes:
        xmin, ymin = reshape_box[0][0] - MARGIN, np.min(np.min(reshape_box[:, 1]))
        xmax, ymax = reshape_box[-1][4], np.max(np.max(reshape_box[:, -1]))

        line_boxes.append([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])

    return line_boxes


def check_area_inside_contour(area=None, contour=None):
    four_edges = [[True, True], [False, True], [False, False], [True, False]]
    contour, area, idx = np.array(contour), np.array(area), 0
    area_center = np.mean(area, axis=0)

    areas_inside_contour = list()
    neg = 0
    pos = 0
    thresholds = [neg, pos]
    for i in range(len(contour)):
        for edge in four_edges:
            for threshold in thresholds:
                flag = ((contour[i][idx] - area_center <= threshold) == edge).all()
                if ((contour[i][idx] - area_center == 0) == [True, True]).all():
                    flag = True
                if flag: break
            areas_inside_contour.append(flag)
            idx += 1

        if np.array(areas_inside_contour).all():
            return np.append(contour[i], area).tolist()
        areas_inside_contour[:] = []
        idx = 0

    return False


def bubbles_sort(arr):
    for idx1 in range(len(arr) - 1):
        for idx2 in range(len(arr) - idx1 - 1):
            if arr[idx2][0] > arr[idx2 + 1][0]:
                arr[idx2], arr[idx2 + 1] = arr[idx2 + 1], arr[idx2]
    return arr


def sort_area_inside_contour(target=None, spacing_word=None):
    re_gt = list()
    for idx in range(len(target)):
        if target[idx] is False: continue
        re_gt.append(target[idx])
    word_re_gt = list()
    for idx in range(len(re_gt)):
        word_re_gt.append(tuple(re_gt[idx][:8]))
    word_re_gt = list(set(word_re_gt))
    tmp = []
    for idx in range(len(word_re_gt)):
        tmp.append(word_re_gt[idx][1])
        word_re_gt[idx] = list(word_re_gt[idx])
    tmp = sorted(tmp)
    for i in range(len(tmp)):
        for j in range(len(word_re_gt)):
            if tmp[i] == word_re_gt[j][1]: word_re_gt.insert(i, word_re_gt.pop(j))

    if spacing_word is not None:
        blank = []
        for od_idx, od in enumerate(spacing_word):
            for z in range(od_idx, len(word_re_gt)):
                if np.array(word_re_gt[z]).astype(int).tolist() == od.tolist(): word_re_gt.insert(od_idx,
                                                                                                  word_re_gt.pop(z))

    final_list = np.array(word_re_gt).copy().tolist()
    for idx1 in range(len(word_re_gt)):
        for idx2 in range(len(re_gt)):
            if word_re_gt[idx1] == re_gt[idx2][:8]:
                final_list[idx1].append(re_gt[idx2][8:])
    final_list = list(final_list)

    if spacing_word is not None:
        for c in range(len(final_list)): blank.append(len(final_list[c][8:]))
        return blank

    final_char_list = np.array([])
    for idx in range(len(final_list)):
        final_char_list = np.append(final_char_list, bubbles_sort(final_list[idx][8:]))
    final_char_list = np.array(final_char_list).reshape(-1, 8).astype(int)

    return final_char_list, word_re_gt


def thresholding(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    return thresh


def warpCoord(Minv, pt):
    out = np.matmul(Minv, (pt[0], pt[1], 1))
    return np.array([out[0] / out[2], out[1] / out[2]])


def labeling(score=None, conectivity=None, map=None, text_thr=None, link_thr=None, FLAGS=None):
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(score.astype(np.uint8),
                                                                         connectivity=conectivity)
    det = []
    img_h, img_w = map.shape
    for k in range(1, nLabels):
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 30: continue

        if np.max(map[labels == k]) < text_thr: continue

        segmap = np.zeros(map.shape, dtype=np.uint8)

        segmap[labels == k] = 255
        if FLAGS == 'word': segmap[np.logical_and(link_thr == 1, text_thr == 0)] = 0

        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1

        if sx < 0: sx = 0
        if sy < 0: sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h

        if FLAGS == 'char':
            kernel_width = int(round(w / 4.5))
            kernel_height = int(round(h / 3.0))
            if kernel_width is 0: kernel_width = 1
            if kernel_height is 0: kernel_height = 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))

        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        np_contours = np.roll(np.array(np.where(segmap != 0)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])

        l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
        t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
        box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)
        [x1,y1], [x2,y1], [x2,y2], [x1,y2] = box[0], box[1], box[2], box[3]
        x1, x2 = x1 + opt.LEFT_CHAR_LINE, x2 + opt.RIGHT_CHAR_LINE
        box = np.array([[x1,y1],[x2,y1], [x2,y2],[x1,y2]])      
        det.append(box)

    return det


def getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text, FLAGS=None):
    linkmap = linkmap.copy()
    textmap = textmap.copy()

    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    if FLAGS == 'char':
        score = text_score
        conectivity = 4

    if FLAGS == 'word':
        ret, link_score = cv2.threshold(linkmap, link_threshold + 0.3, 1, 0)
        score = np.clip(text_score + link_score, 0, 1)
        conectivity = 8

    if FLAGS == 'line':
        ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)
        line_kernel = np.ones((2, opt.LNK_KERNEL_SIZE), np.uint8)
        link_score = cv2.dilate(link_score, line_kernel, iterations=3)
        score = np.clip(text_score + link_score, 0, 1)
        conectivity = 8

    det = labeling(score=score, conectivity=conectivity, map=textmap, text_thr=text_threshold, link_thr=link_threshold,
                   FLAGS=FLAGS)

    return det


def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text):
    boxes = getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text, FLAGS='char')
    word_boxes = getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text, FLAGS='word')
    line_boxes = getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text, FLAGS='line')
    polys = [None] * len(boxes)
    word_polys = [None] * len(word_boxes)
    line_polys = [None] * len(line_boxes)
    return boxes, polys, word_boxes, word_polys, line_boxes, line_polys


def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=2):
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys
