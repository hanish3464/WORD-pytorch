"""test.py"""


# -*- coding: utf-8 -*-

from collections import OrderedDict
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import time
import cv2
import numpy as np
import config
from wtd import WTD
import wtd_utils
import file_utils
import imgproc
import sys

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


def test_net(net, image, text_threshold, link_threshold, low_text, cuda):
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, config.MAXIMUM_IMAGE_SIZE, interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=config.MAG_RATIO)
    ratio_h = ratio_w = 1 / target_ratio

    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))

    if cuda: x = x.cuda()

    y, _ = net(x)

    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    boxes, polys, word_boxes, word_polys, line_boxes, line_polys = wtd_utils.getDetBoxes(score_text, score_link,
                                                                                         text_threshold, link_threshold,
                                                                                         low_text)

    boxes = wtd_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = wtd_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)

    word_boxes = wtd_utils.adjustResultCoordinates(word_boxes, ratio_w, ratio_h)
    word_polys = wtd_utils.adjustResultCoordinates(word_polys, ratio_w, ratio_h)

    line_boxes = wtd_utils.adjustResultCoordinates(line_boxes, ratio_w, ratio_h)
    line_polys = wtd_utils.adjustResultCoordinates(line_polys, ratio_w, ratio_h)

    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]
    for a in range(len(word_polys)):
        if word_polys[a] is None: word_polys[a] = word_boxes[a]
    for l in range(len(line_polys)):
        if line_polys[l] is None: line_polys[l] = line_boxes[l]

    return polys, word_polys, line_polys, score_text


def test ():
    ''''''

    '''INITIALIZE MODEL AND LOAD PRETRAINED MODEL'''
    myNet = WTD()
    print('Loading model from defined path :'  + config.PRETRAINED_MODEL_PATH)
    if config.cuda:
        myNet.load_state_dict(copyStateDict(torch.load(config.PRETRAINED_MODEL_PATH)))
    else:
        myNet.load_state_dict(copyStateDict(torch.load(config.PRETRAINED_MODEL_PATH, map_location='cpu')))

    if config.cuda:
        myNet = myNet.cuda()
        myNet = torch.nn.DataParallel(myNet)
        cudnn.benchmark = False

    spacing_word = []
    myNet.eval()
    t = time.time()

    ''' SET PATH '''
    #DEFAULT_PATH_LIST = [config.TEST_IMAGE_PATH, config.TEST_PREDICTION_PATH, config.CANVAS_PATH, config.MASK_PATH,
    #                    config.BBOX_PATH, config.RESULT_CHAR_PATH, config.SPACING_WORD_PATH]
    DEFAULT_PATH_LIST = [config.TEST_IMAGE_PATH, config.TEST_PREDICTION_PATH, config.RESULT_CHAR_PATH, config.SPACING_WORD_PATH]
    for PATH in DEFAULT_PATH_LIST:
        if not os.path.isdir(PATH): os.mkdir(PATH)

    ''' LIST IMAGE FILE '''
    img_list, _,_, name_list = file_utils.get_files(config.TEST_IMAGE_PATH)


    ''' KICK OFF TEST PROCESS '''
    for i, img in enumerate(img_list):

        sys.stdout.write('TEST IMAGES: {:d}/{:d}: {:s} \r'.format(i + 1, len(img_list), img))
        sys.stdout.flush()

        ''' LOAD IMAGE '''
        img = imgproc.loadImage(img)
        img = imgproc.delNoiseBorderLine(img)
        ''' ADJUST IMAGE SIZE AND MAKE BORDER LINE FOR BETTER TESTING ACCURACY '''
        constant = imgproc.adjustImageRatio(img)
        index1 = name_list[i]
        copy_img = constant.copy()

        ''' PASS THE TEST MODEL AND PREDICT BELOW 4 RESULTS '''
        charBBoxes, wordBBoxes, lineBBoxes, heatmap = test_net(myNet, constant, config.text_threshold,
                                                               config.link_threshold, config.low_text, config.cuda)

        #file_utils.saveImage(dir=config.CANVAS_PATH, img=constant, index1=index1)
        #file_utils.saveMask(dir=config.MASK_PATH, heatmap=heatmap, index1=index1)

        chars_inside_line = []
        words_inside_line = []
        chars_inside_word = []

        ''' CHECK THERE IS INER BBOX IN OUTER BBOX '''
        for a in range(len(charBBoxes)):
            chars_inside_line.append(wtd_utils.checkAreaInsideContour(area=charBBoxes[a], contour=lineBBoxes))
        for b in range(len(wordBBoxes)):
            words_inside_line.append(wtd_utils.checkAreaInsideContour(area=wordBBoxes[b], contour=lineBBoxes))
        for c in range(len(charBBoxes)):
            chars_inside_word.append(wtd_utils.checkAreaInsideContour(area=charBBoxes[c], contour=wordBBoxes))

        '''INNER BBOX SORTING'''
        charBBoxes, lineBBoxes = wtd_utils.sortAreaInsideContour(target=chars_inside_line, spacing_word=None)
        wordBBoxes, lineBBoxes = wtd_utils.sortAreaInsideContour(target=words_inside_line, spacing_word=None)
        count = wtd_utils.sortAreaInsideContour(target=chars_inside_word, spacing_word=wordBBoxes)
        spacing_word.append(count)

        tmp_charBBoxes = np.array(charBBoxes, dtype=np.float32).reshape(-1, 4, 2).copy()


        '''DRAW BBOX ON IMAGE'''
        #file_utils.drawBBoxOnImage(dir=config.BBOX_PATH, img=copy_img2, index1=index1, boxes=charBBoxes, flags='char')
        #file_utils.drawBBoxOnImage(dir=config.BBOX_PATH, img=copy_img2, index1=index1, boxes=wordBBoxes, flags='word')
        #file_utils.drawBBoxOnImage(dir=config.BBOX_PATH, img=copy_img2, index1=index1, boxes=lineBBoxes, flags='line')


        '''MAKE FINAL CHARACTER IMAGE FOR RECOGNITION PROCESS'''
        for j, charBBox in enumerate(tmp_charBBoxes):
            index2 = file_utils.adjustImageNum(j, len(tmp_charBBoxes))
            char = imgproc.cropBBoxOnImage(copy_img, charBBox)
            orig_char = imgproc.adjustImageBorder(char, img_size=config.recognition_input_size, color=config.white)
            thresh_char = wtd_utils.thresholding(orig_char, img_size=config.recognition_input_size)
            #file_utils.saveImage(dir=config.RESULT_CHAR_PATH, img=orig_char, index1=index1, index2=index2)
            file_utils.saveImage(dir=config.RESULT_CHAR_PATH, img=thresh_char, index1=index1, index2=index2)

    ''' GENERATE TEXT FILE FOR SPACEING WORD '''
    file_utils.saveText(dir=config.SPACING_WORD_PATH, text=spacing_word, index1='spacing_word')

    print("TOTAL TIME : {}s".format(time.time() - t))
