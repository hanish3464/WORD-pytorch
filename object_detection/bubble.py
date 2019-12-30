# -*- coding: utf-8 -*-

import sys
import numpy as np
import torch
from torch.autograd import Variable
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
import object_detection.bubble_utils as bubble_utils
import os
import opt

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


def test_net(model=None, image=None, params=None, bg=None, cls=None):
    blob, scale, label = params
    with torch.no_grad():  # pre-processing data for passing net
        im_data = Variable(torch.FloatTensor(1).cuda())
        im_info = Variable(torch.FloatTensor(1).cuda())
        num_boxes = Variable(torch.LongTensor(1).cuda())
        gt_boxes = Variable(torch.FloatTensor(1).cuda())

    im_info_np = np.array([[blob.shape[1], blob.shape[2], scale[0]]], dtype=np.float32)
    im_data_pt = torch.from_numpy(blob)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    with torch.no_grad():  # resize
        im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.resize_(1, 1, 5).zero_()
        num_boxes.resize_(1).zero_()

    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label = model(im_data, im_info, gt_boxes, num_boxes)  # predict

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]

    if opt.TEST_BBOX_REG:
        box_deltas = bbox_pred.data
        if opt.TRAIN_BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            if opt.cuda:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(opt.TRAIN_BBOX_NORMALIZE_STDS).cuda() \
                             + torch.FloatTensor(opt.TRAIN_BBOX_NORMALIZE_MEANS).cuda()
            else:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(opt.TRAIN_BBOX_NORMALIZE_STDS) \
                             + torch.FloatTensor(opt.TRAIN_BBOX_NORMALIZE_MEANS)

            box_deltas = box_deltas.view(1, -1, 4 * len(label))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

    pred_boxes /= scale[0]

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()

    image = np.copy(image[:, :, ::-1])
    demo = image.copy()
    bubbles = []
    dets_bubbles = []

    for j in range(1, len(label)):
        inds = torch.nonzero(scores[:, j] > opt.THRESH).view(-1)
        if inds.numel() > 0:
            cls_scores = scores[:, j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_boxes[order, :], cls_scores[order], opt.TEST_NMS)
            cls_dets = cls_dets[keep.view(-1).long()].cpu().numpy()

            #  post-processing : get contours of speech bubble
            demo, image, bubbles, dets_bubbles = bubble_utils.get_cnt_bubble(image, image.copy(), label[j], cls_dets,
                                                                             cls, bg=bg)
    return demo, image, bubbles, dets_bubbles
