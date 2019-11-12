import cv2
import numpy as np

def maskSpeechBubble(img, dets):

    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        crop = img[ymin:ymax, xmin:xmax, :]
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray_crop, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        maxIdx = 0
        maxArea = 0
        for x, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if maxArea < area:
                maxArea = area
                maxIdx = x
        h, w, _ = crop.shape

        cv2.drawContours(crop, [contours[maxIdx]], 0, (255, 150, 150), -1)
        img[ymin:ymax, xmin:xmax, :] = crop[:, :, :]

    return img


