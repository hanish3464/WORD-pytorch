"""test.py"""

# -*- coding: utf-8 -*-

import _init_paths
import sys
import numpy as np
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.faster_rcnn.resnet import resnet
from model.utils.config import cfg
import file_utils
import imgproc
import config
import sbd_utils
import text
import cv2
import os


def test_net(fasterRCNN, image, img_blob, img_scales, items, labels, i):
    im_data, im_info, num_boxes, gt_boxes = items
    im_info_np = np.array([[img_blob.shape[1], img_blob.shape[2], img_scales[0]]], dtype=np.float32)
    im_data_pt = torch.from_numpy(img_blob)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    with torch.no_grad():
        im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.resize_(1, 1, 5).zero_()
        num_boxes.resize_(1).zero_()

    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]

    if config.TEST_BBOX_REG:
        box_deltas = bbox_pred.data
        if config.TRAIN_BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            if config.cuda:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(config.TRAIN_BBOX_NORMALIZE_STDS).cuda() \
                             + torch.FloatTensor(config.TRAIN_BBOX_NORMALIZE_MEANS).cuda()
            else:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(config.TRAIN_BBOX_NORMALIZE_STDS) \
                             + torch.FloatTensor(config.TRAIN_BBOX_NORMALIZE_MEANS)

            box_deltas = box_deltas.view(1, -1, 4 * len(labels))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

    pred_boxes /= img_scales[0]

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()

    copy_img = np.copy(image[:, :, ::-1])
    bubbles = []
    for j in range(1, len(labels)):
        inds = torch.nonzero(scores[:, j] > config.THRESH).view(-1)
        if inds.numel() > 0:
            cls_scores = scores[:, j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_boxes[order, :], cls_scores[order], config.TEST_NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            copy_img, vis_img, bubbles = sbd_utils.divideBubbleFromImage(copy_img, image[:, :, ::-1], labels[j],
                                                                         cls_dets.cpu().numpy(),
                                                                         config.CLASS_THRESH, bg=config.BACKGROUND)

    copy_img, vis_img, cuts, rect_cuts = sbd_utils.divideCutFromImage(copy_img, image[:, :, ::-1], i,
                                                                      bg=config.BACKGROUND)
    # alpha_image = sbd_utils.addImageToAlphaChannel(copy_img, copy_img, FLAG='conversion')
    # vis_img, texts = text.detection(vis_img, bubbles, boxes)
    return cuts, bubbles, vis_img, rect_cuts
    # return alpha_image, vis_img, cuts, bubbles, texts


def test():
    ''' '''

    DEFAULT_PATH_LIST = [config.TEST_PREDICTION_PATH, config.CUT_PATH, config.BUBBLE_PATH, config.FINAL_IMAGE_PATH,
                         config.RECT_CUT_PATH]
    for PATH in DEFAULT_PATH_LIST:
        if not os.path.isdir(PATH): os.mkdir(PATH)

    np.random.seed(config.RNG_SEED)
    labels = np.asarray(['__background__', 'speech'])

    '''INITIALIZE MODEL AND LOAD PRETRAINED MODEL'''
    layerNum = 101
    if config.BACKBONE == 'res152': layerNum = 152

    fasterRCNN = resnet(labels, layerNum, pretrained=False, class_agnostic=False)
    fasterRCNN.create_architecture()

    print('Loading model from defined path :' + config.PRETRAINED_MODEL_PATH)

    if config.cuda:
        model = torch.load(config.PRETRAINED_MODEL_PATH)
    else:
        model = torch.load(config.PRETRAINED_MODEL_PATH, map_location=(lambda storage, loc: storage))

    fasterRCNN.load_state_dict(model['model'])
    if 'pooling_mode' in model.keys(): cfg.POOLING_MODE = model['pooling_mode']

    if config.cuda:
        fasterRCNN.cuda()
        # fasterRCNN = torch.nn.DataParallel(fasterRCNN)

    fasterRCNN.eval()
    t = time.time()

    with torch.no_grad():
        im_data = Variable(torch.FloatTensor(1).cuda())
        im_info = Variable(torch.FloatTensor(1).cuda())
        num_boxes = Variable(torch.LongTensor(1).cuda())
        gt_boxes = Variable(torch.FloatTensor(1).cuda())
        items = [im_data, im_info, num_boxes, gt_boxes]

    ''' LIST IMAGE FILE '''
    img_list, _, _ = file_utils.get_files(config.TEST_IMAGE_PATH)

    ''' KICK OFF TEST PROCESS '''
    for i, img in enumerate(img_list):
        sys.stdout.write('TEST IMAGES: {:d}/{:d}: {:s} \r'.format(i + 1, len(img_list), img))
        sys.stdout.flush()

        ''' LOAD IMAGE '''
        img = imgproc.loadImage(img)
        img_blob, img_scales = imgproc.getImageBlob(img)

        ''' PASS THE TEST MODEL AND PREDICT BELOW IM RESULTS '''
        # alpha_img, vis_img, cuts, bubbles, texts = test_net(fasterRCNN, img, img_blob, img_scales, items, labels, i)
        cuts, bubbles, vis_img, rect_cuts = test_net(fasterRCNN, img, img_blob, img_scales, items, labels, i)
        fixed_i = file_utils.resultNameNumbering(origin=i, digit=len(img_list))

        for cut_idx, cut in enumerate(cuts):
            file_utils.saveImage(dir=config.CUT_PATH, img=cut, index1=fixed_i, index2=cut_idx, ext='.png')
        for bub_idx, bubble in enumerate(bubbles):
            file_utils.saveImage(dir=config.BUBBLE_PATH, img=bubble, index1=fixed_i, index2=bub_idx, ext='.png')
        for rect_cut_idx, rect_cut in enumerate(rect_cuts):
            file_utils.saveImage(dir=config.RECT_CUT_PATH, img=rect_cut, index1=fixed_i, index2=rect_cut_idx,
                                 ext='.png')

        # for txt_idx, txt in enumerate(texts):
        #     file_utils.saveImage(dir=config.TEXT_PATH, img=txt, index1=fixed_i, index2=txt_idx, ext='.png')

        file_utils.saveImage(dir=config.FINAL_IMAGE_PATH, img=vis_img, index1=fixed_i, ext='.jpg')

    print("TOTAL TIME : {}s".format(time.time() - t))
