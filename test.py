"""test.py"""


# -*- coding: utf-8 -*-

from collections import OrderedDict
import os
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import time
import cv2
import numpy as np

import config
from wtd import WTD
import postprocess
import preprocess
import debug
import file

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

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly):
    t0 = time.time()

    # RESIZE IMAGE
    img_resized, target_ratio, size_heatmap = preprocess.resize_aspect_ratio(image, config.image_size, interpolation=cv2.INTER_LINEAR, mag_ratio=config.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    """PREPROCESSING"""
    x = preprocess.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # HWC to CHW
    x = Variable(x.unsqueeze(0))                # CHW to BCHW

    """GPU"""
    if cuda:
        x = x.cuda()

    """PRETRAINED MODEL PREDICTION with FORWARD"""
    y, _ = net(x)

    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    #debug.printing(score_text)
    #debug.printing(score_link)

    t0 = time.time() - t0
    t1 = time.time()


    #POSTPROCESSING
    boxes, polys = postprocess.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    #COORDINATE ADJUSTMENT
    boxes = postprocess.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = postprocess.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    #RENDER RESULTS(optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = preprocess.cvt2HeatmapImg(render_img)
    debug.printing(ret_score_text)
    if config.show_time : print("\nPOST PRECESSING TIME : {:.3f}/{:.3f}".format(t0, t1))
    return boxes, polys, ret_score_text

def test ():

    #MODEL INITIALIZE
    myNet = WTD()
    print('Loading model from defined path :'  + config.pretrained_model_path)
    if config.cuda:#GPU
        myNet.load_state_dict(copyStateDict(torch.load(config.pretrained_model_path)))

    else:#ONLY CPU
        myNet.load_state_dict(copyStateDict(torch.load(config.pretrained_model_path, map_location='cpu')))

    if config.cuda:
        myNet = myNet.cuda()
        myNet = torch.nn.DataParallel(myNet)
        cudnn.benchmark = False

    myNet.eval()
    t = time.time()

    image_list, _,_ = file.get_files(config.test_images_folder_path)

    if not os.path.isdir(config.prediction_folder):
        os.mkdir(config.prediction_folder)
    if not os.path.isdir(config.mask_folder):
        os.mkdir(config.mask_folder)
    if not os.path.isdir(config.ground_truth_folder):
        os.mkdir(config.ground_truth_folder)

    for k, image_path in enumerate(image_list):
        print("TEST IMAGE: {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path))
        image = preprocess.loadImage(image_path)

        bboxes, polys, score_text = test_net(myNet, image, config.text_threshold, config.link_threshold, config.low_text, config.cuda, config.poly)

        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = config.mask_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        file.saveResult(image_path, image[:,:,::-1], polys, dir1=config.prediction_folder, dir2=config.ground_truth_folder)
    print("TOTAL TIME : {}s".format(time.time() - t))
