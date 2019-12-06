import cv2
import numpy as np
import math
import config
from text_detection import test, file_utils, imgproc, wtd_utils
from text_detection import config as cfg


def text(model, vis_img, dets_bub, bubbles):
    spacing_word = []
    cnt = 0

    for det_bub, bubble in zip(dets_bub, bubbles):
        cnt += 1
        bubble = imgproc.delNoiseBorderLine(bubble)
        constant = imgproc.adjustImageRatio(bubble)
        charBBoxes, wordBBoxes, lineBBoxes, heatmap = test.test_net(model, constant, cfg.text_threshold,
                                                                    cfg.link_threshold, cfg.low_text, cfg.cuda)

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
        xmin, ymin, xmax, ymax = det_bub
        file_utils.drawBBoxOnImage(dir=config.TEXT_PATH, img=vis_img[ymin:ymax, xmin:xmax, :], index1=str(cnt),
                                          boxes=charBBoxes, flags='char')

    return vis_img
