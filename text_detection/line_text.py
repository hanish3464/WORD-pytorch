import numpy as np
import opt
import imgproc
import torch
import cv2
import file_utils
from torch.autograd import Variable
import text_detection.ltd_utils as ltd_utils
import os
#from text_detection import test, file_utils, imgproc, wtd_utils
#from text_detection import config as cfg

def test_net(net, image, text_threshold, link_threshold, low_text, cuda):
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, opt.MAXIMUM_IMAGE_SIZE, interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=opt.MAG_RATIO)
    ratio_h = ratio_w = 1 / target_ratio

    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))

    if cuda: x = x.cuda()

    y, _ = net(x)

    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    boxes, polys, word_boxes, word_polys, line_boxes, line_polys = ltd_utils.getDetBoxes(score_text, score_link,
                                                                                         text_threshold, link_threshold,
                                                                                         low_text)

    boxes = ltd_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = ltd_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)

    word_boxes = ltd_utils.adjustResultCoordinates(word_boxes, ratio_w, ratio_h)
    word_polys = ltd_utils.adjustResultCoordinates(word_polys, ratio_w, ratio_h)

    line_boxes = ltd_utils.adjustResultCoordinates(line_boxes, ratio_w, ratio_h)
    line_polys = ltd_utils.adjustResultCoordinates(line_polys, ratio_w, ratio_h)

    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]
    for a in range(len(word_polys)):
        if word_polys[a] is None: word_polys[a] = word_boxes[a]
    for l in range(len(line_polys)):
        if line_polys[l] is None: line_polys[l] = line_boxes[l]

    return polys, word_polys, line_polys, score_text


def test(model=None, demo=None, bubbles=None, dets=None, img_name=None, save_to=None):
    spacing_word = []
    bubble_order_num = 0
    demo_warp_items = []
    eng_bubbles = []

    for det, bubble in zip(dets, bubbles):
        alpha_bubble = imgproc.cpImage(bubble)
        name_final = img_name + '_' + str(bubble_order_num)
        bubble = imgproc.delNoiseBorderLine(bubble)
        constant = imgproc.adjustImageRatio(bubble)
        charBBoxes, wordBBoxes, lineBBoxes, heatmap = test_net(model, constant, opt.text_threshold,
                                                                    opt.link_threshold, opt.low_text, opt.cuda)
        copy_img = constant.copy()
        chars_inside_line = []
        words_inside_line = []
        chars_inside_word = []

        ''' CHECK THERE IS INER BBOX IN OUTER BBOX '''
        for a in range(len(charBBoxes)):
            chars_inside_line.append(ltd_utils.checkAreaInsideContour(area=charBBoxes[a], contour=lineBBoxes))
        for b in range(len(wordBBoxes)):
            words_inside_line.append(ltd_utils.checkAreaInsideContour(area=wordBBoxes[b], contour=lineBBoxes))
        for c in range(len(charBBoxes)):
            chars_inside_word.append(ltd_utils.checkAreaInsideContour(area=charBBoxes[c], contour=wordBBoxes))
        
        '''INNER BBOX SORTING'''
        charBBoxes, lineBBoxes = ltd_utils.sortAreaInsideContour(target=chars_inside_line, spacing_word=None)
        wordBBoxes, lineBBoxes = ltd_utils.sortAreaInsideContour(target=words_inside_line, spacing_word=None)
        count = ltd_utils.sortAreaInsideContour(target=chars_inside_word, spacing_word=wordBBoxes)
        spacing_word.append(count)

        lineBBoxes = ltd_utils.link_refine(boxes=charBBoxes, MARGIN=opt.MARGIN)
        xmin, ymin, xmax, ymax = det
        demo_warp_items.append([demo[ymin:ymax, xmin:xmax, :], lineBBoxes])
        #eng_bubbles.append([alpha_bubble, lineBBoxes])

        '''MAKE FINAL CHARACTER IMAGE FOR RECOGNITION PROCESS'''
        tmp_charBBoxes = np.array(charBBoxes, dtype=np.float32).reshape(-1, 4, 2).copy()
        for j, charBBox in enumerate(tmp_charBBoxes):
            index2 = file_utils.resultNameNumbering(j, len(tmp_charBBoxes))
            char = imgproc.cropBBoxOnImage(copy_img, charBBox)
            orig_char = imgproc.adjustImageBorder(char, img_size=opt.recognition_input_size, color=opt.white)
            thresh_char = ltd_utils.thresholding(orig_char, img_size=opt.recognition_input_size)
            file_utils.saveImage(save_to=save_to, img=thresh_char, index1=name_final, index2=index2, ext='.png')
        bubble_order_num += 1
        name_final = ''
    return demo, spacing_word, demo_warp_items
