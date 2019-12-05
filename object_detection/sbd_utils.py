import cv2
import numpy as np
import config


def addImageToAlphaChannel(canvas, img, FLAG=None):
    b_channel, g_channel, r_channel = cv2.split(img)
    if FLAG == 'segmentation':
        img_BGRA = cv2.merge((b_channel, g_channel, r_channel, canvas))
    else:
        alpha = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
        img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha))
    return img_BGRA


def bubbleAlphaBlending(img, contours, index):
    h, w, c = img.shape
    canvas = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(canvas, [contours[index]], 0, (255, 255, 255), -1)
    alpha_img = addImageToAlphaChannel(canvas, img, FLAG='segmentation')

    return alpha_img


def drawBubbleContours(img, class_name, score, new_contours, maxIdx, idx):
    if config.DRAWBUB is True:
        cv2.drawContours(img, [new_contours[maxIdx]], 0, (255, 0, 0), 3)
        cv2.putText(img, "{:s}_{:d}: {:.3f}".format(class_name, idx, score), (0, 10), cv2.FONT_HERSHEY_PLAIN,
                1.0, (255, 0, 0), thickness=2)
    return img


def divideBubbleFromImage(img, vis_img, class_name, dets, class_thresh=0.8, bg='white'):

    count = 0
    bubble_list = []
    value = []
    bboxes = []
    scores = []

    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > class_thresh:
            scores.append(score)
            ymin = int(bbox[1])
            value.append(ymin)
            bboxes.append(bbox)

    order_k = sorted(range(len(value)), key=lambda k: value[k])

    for bub_order in order_k:

            bbox = bboxes[bub_order]
            xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            crop = img[ymin:ymax, xmin:xmax, :]
            crop_tmp = crop.copy()
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray_crop, 127, 255, cv2.THRESH_BINARY)
            _, contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

            maxIdx = 0
            maxArea = 0
            for x, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if maxArea < area:
                    maxArea = area
                    maxIdx = x

            count += 1
            crop_tmp = drawBubbleContours(crop_tmp, class_name, scores[bub_order], contours, maxIdx, count)
            vis_img[ymin:ymax, xmin:xmax, :] = crop_tmp[:, :, :]
            h, w, _ = crop.shape
            bubble = bubbleAlphaBlending(crop, contours, maxIdx)

            bubble_list.append(bubble)
            if bg == 'white':
                cv2.drawContours(crop, [contours[maxIdx]], 0, (255, 255, 255), -1)
                cv2.drawContours(crop, [contours[maxIdx]], 0, (255, 255, 255), 10)

            if bg == 'black':
                cv2.drawContours(crop, [contours[maxIdx]], 0, (0, 0, 0), -1)
                cv2.drawContours(crop, [contours[maxIdx]], 0, (0, 0, 0), 10)

    return img, vis_img, bubble_list


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
        cuts.append(orig[y:y+h, x:x+w, :])
        if config.DRAWCUT is True:
            cv2.drawContours(img, [new_hull], 0, (0, 0, 255), 3)
            cv2.putText(img, 'Cut_' + str(idx), (x + w - 80, y - 5), cv2.FONT_HERSHEY_PLAIN,
                    1.5, (0, 0, 255), thickness=2)
    return img, cuts


def cutAlphaBlending(img, new_contours):
    cut_list = []; det = []
    for idx, new_cnt in enumerate(new_contours):
        x, y, w, h = cv2.boundingRect(new_cnt)
        img_h, img_w, _ = img.shape
        canvas = np.zeros((img_h, img_w), dtype=np.uint8)
        new_hull = cv2.convexHull(new_cnt, clockwise=True)
        cv2.drawContours(canvas, [new_hull], 0, (255, 255, 255), -1)
        canvas_crop = canvas[y:y+h, x:x+w]
        img_crop = img[y:y+h, x:x+w]
        alpha_img = addImageToAlphaChannel(canvas_crop, img_crop, FLAG='segmentation')
        cut_list.append(alpha_img)
        det.append([x,y,x+w,y+h])
    return cut_list, det


def divideCutFromImage(img, vis_img, idx, bg='white'):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if bg == 'black':
        _, thresh = cv2.threshold(gray, 0.1, 255, 0)
    elif bg == 'white':
        _, thresh = cv2.threshold(gray, config.THRESH_EXTENT, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_not(thresh)
    else:
        _, thresh = cv2.threshold(gray, config.THRESH_EXTENT, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_not(thresh)

    kernel = np.ones((3, 3), np.uint8)
    if bg == 'black':
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=config.OPENING_ITER_NUM)

    if bg == 'white' or bg == 'black':
        thresh = cv2.dilate(thresh, kernel, iterations=config.DILATE_ITER_NUM)
    if bg == 'classic':
        thresh = cv2.copyMakeBorder(thresh[:, 10:-10], 0, 0, 10, 10, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    hull_lists = []
    img_h, img_w, _ = np.array(img).shape
    canvas = np.zeros((img_h, img_w), dtype=np.uint8)
    for i in contours:
        area = cv2.contourArea(i)
        hull = cv2.convexHull(i, clockwise=True)
        if bg == 'white' and area < config.AREA_THRESH: continue
        hull_lists.append(hull)
    new_contours = removeNoiseConvexHull(canvas, hull_lists)
    cut_list, det = cutAlphaBlending(img, new_contours)
    vis_img, rect_cuts = drawCutConvexHull(img, vis_img, new_contours)

    return img, vis_img, cut_list, rect_cuts