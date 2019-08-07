"""
This Project is WTD(Webtoon Text Detection) based on NAVER CLOVA AI RESEARCH paper.

Future Science Technology Internship
Ajou Univ.
Major : Software and Computer Engineering
Writer: Han Kim

"""
# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import postprocess
import preprocess
import file
import evaluation
import json
import zipfile
import config
from wtd import WTD

from collections import OrderedDict

#debug function
import debug

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

#ARGUMENT PARSER START
parser = argparse.ArgumentParser(description='Webtoon Text Localization(Detection)')

parser.add_argument('--train', default=False, type=bool, help='train flag')
parser.add_argument('--test', default=False, type=bool, help='test flag')
parser.add_argument('--evaluation', default=False, type=bool, help='evaluation flag')



args = parser.parse_args()

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly):
    t0 = time.time()

    # RESIZE IMAGE
    img_resized, target_ratio, size_heatmap = preprocess.resize_aspect_ratio(image, config.image_size, interpolation=cv2.INTER_LINEAR, mag_ratio=config.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    """PREPROCESSING"""
    x = preprocess.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # HWC to CHW
    x = Variable(x.unsqueeze(0))                # CHW to BCHW ex) 1,3,2496,512 test is batch 1
                                                # but training batch depends on the number of GPUs 1 per 8 batch

    """GPU"""
    if cuda:
        x = x.cuda()

    """PRETRAINED MODEL PREDICTION with FORWARD"""
    """x is normarized state and BCHW format. It is necessary condition for CNN training"""
    y, _ = net(x)
    """Don't be surprised about predicted result. net is already trained with synthetic images"""
    """so, you can just net is good at fining character regions """

    """MAKE SCORE AND LINK MAP SCORE=CHR, LINK=AFFINITY"""
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()
    #this score is written as heatmap format. Net predicts heatmap format.
    debug.printing(score_text)
    debug.printing(score_link)
    t0 = time.time() - t0
    t1 = time.time()


    #POSTPROCESSING
    #heatmap is changed to coordinate after postpp
    boxes, polys = postprocess.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    #box shape returns

    #COORDINATE ADJUSTMENT
    #box size should be magnified because image size is magnified
    boxes = postprocess.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = postprocess.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]
    #if images have polys, it was stored in polys[], else box is also stored in polys[]
    #so, polys have all bounding boxes coordinates in images.
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
        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = config.mask_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        #image[:,:,::-1] is to change RGB to BGR / final ::-1 means inverse order of channel (RGB->BGR)
        #so, image color is changed as blue tone.
        file.saveResult(image_path, image[:,:,::-1], polys, dir1=config.prediction_folder, dir2=config.ground_truth_folder)
    print("TOTAL TIME : {}s".format(time.time() - t))

def train():
    pass

if __name__ == '__main__':
    if args.train:
        train()
    if args.test:
         test()
    if args.evaluation:
        evaluation.evaluation()
