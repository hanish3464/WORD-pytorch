import cv2
import numpy as np
import math
import config


def findTextRegions(thr, BOX_SIZE_THRESHOLD):
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thr.astype(np.uint8), connectivity=4)
    img_h, img_w = thr.shape
    det = []
    for k in range(1, nLabels):
        size = stats[k, cv2.CC_STAT_AREA]
        if size < BOX_SIZE_THRESHOLD: continue
        if np.max(thr[labels == k]) < 0.7: continue

        segmap = np.zeros(thr.shape, dtype=np.uint8)
        segmap[labels == k] = 255

        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1

        if sx < 0: sx = 0
        if sy < 0: sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        np_contours = np.roll(np.array(np.where(segmap != 0)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)
        det.append(box)

    return det


def drawTextBBoxes(img, dets, box):
    xmin, ymin, xmax, ymax = box
    crops = []
    for k, det in enumerate(dets):
        x1, y1 = det[0]
        x2, y2 = det[2]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if k == 0:
            if config.DRAWTXT is True:
                cv2.putText(img[ymin:ymax, xmin:xmax, :], "{:s}".format('text'), (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 255, 0), thickness=2)
        if config.DRAWTXT is True:
            cv2.rectangle(img[ymin:ymax, xmin:xmax, :], (x1 + config.TEXT_MIN_MARGIN, y1),
                      (x2 - config.TEXT_MAX_MARGIN, y2), (0, 255, 0), 2)
        crops.append(img[ymin:ymax, xmin:xmax, :])
    return img, crops


def delNoiseBorderLine(img):
    canvas = np.zeros(img.shape, dtype=img.dtype)
    canvas[img[:, :, 3] == 255] = 255
    kernel = np.ones((3, 3), np.uint8)
    canvas = cv2.erode(canvas, kernel, iterations=4)
    img[canvas == 0] = 255
    return img


def detection(vis_img, bubbles, boxes):
    DILATE_ITER_NUM = 2
    BOX_SIZE_THRESHOLD = 300
    texts = []
    for k, bubble in enumerate(bubbles):
        img = delNoiseBorderLine(bubble)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, 800, 1000)
        _, thr = cv2.threshold(edge, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thr, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        linking_kernel = np.ones((1, config.KERNEL_SIZE), np.uint8)
        thr = cv2.dilate(thr, linking_kernel, iterations=DILATE_ITER_NUM)
        dets = findTextRegions(thr, BOX_SIZE_THRESHOLD)
        vis_img, texts = drawTextBBoxes(vis_img, dets, boxes[k])

    return vis_img, texts


if __name__ == '__main__':
    detection()
