import cv2
import numpy as np
import opt
import imgproc


def removeNoiseConvexHull(canvas, hull_lists):
    for hull in hull_lists: cv2.drawContours(canvas, [hull], 0, (255, 255, 255), -1)
    kernel = np.ones((7, 7), dtype=np.uint8)
    canvas = cv2.erode(canvas, kernel, iterations=1)
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_OPEN, kernel, iterations=5)
    _, contours, _ = cv2.findContours(canvas, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    for idx, i in enumerate(contours):
        hull = cv2.convexHull(i, clockwise=True)
        cv2.drawContours(canvas, [hull], 0, (255, 255, 255), -1)
    _, contours, _ = cv2.findContours(canvas, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    return contours


def drawCutConvexHull(orig, img, new_contours):
    img = np.array(img)
    cuts = []
    for idx, new_cnt in enumerate(reversed(new_contours)):
        new_hull = cv2.convexHull(new_cnt, clockwise=True)
        x, y, w, h = cv2.boundingRect(new_cnt)
        cuts.append(orig[y:y + h, x:x + w, :])
        if opt.DRAWCUT is True:
            cv2.drawContours(img, [new_hull], 0, (0, 0, 255), 3)
            cv2.putText(img, 'Cut_' + str(idx+1), (x + w - 80, y - 5), cv2.FONT_HERSHEY_PLAIN,
                        1.5, (0, 0, 255), thickness=2)
    return img, cuts


def cutAlphaBlending(img, new_contours):
    cut_list = []
    det = []
    for idx, new_cnt in enumerate(new_contours):
        x, y, w, h = cv2.boundingRect(new_cnt)
        img_h, img_w, _ = img.shape
        canvas = np.zeros((img_h, img_w), dtype=np.uint8)
        new_hull = cv2.convexHull(new_cnt, clockwise=True)
        cv2.drawContours(canvas, [new_hull], 0, (255, 255, 255), -1)
        canvas_crop = canvas[y:y + h, x:x + w]
        img_crop = img[y:y + h, x:x + w]
        alpha_img = imgproc.addImageToAlphaChannel(canvas_crop, img_crop, FLAG='segmentation')
        cut_list.append(alpha_img)
        det.append([x, y, x + w, y + h])
    return cut_list, det

def sortCut(dets):
    tmp = []; value = []; dets_cut = []
    for idx, det in enumerate(dets):
        ymin = int(det[1])
        value.append(ymin)
        tmp.append(det)
    order_k = sorted(range(len(value)), key=lambda k: value[k])
    for cut_order in order_k:
        bbox = tmp[cut_order]
        xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        dets_cut.append([xmin,ymin,xmax,ymax])
    return dets_cut