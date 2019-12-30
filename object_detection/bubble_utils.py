import cv2
import numpy as np
import opt
import imgproc


def alpha_blend_bubble(img, contours, index):
    h, w, c = img.shape
    canvas = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(canvas, [contours[index]], 0, (255, 255, 255), -1)
    alpha_img = imgproc.addImageToAlphaChannel(canvas, img, FLAG='segmentation')

    return alpha_img


def draw_bubble_contour(img, class_name, score, new_contours, maxIdx, idx):
    if opt.DRAWBUB is True:
        cv2.drawContours(img, [new_contours[maxIdx]], 0, (255, 0, 0), 3)
        cv2.putText(img, "{:s}_{:d}: {:.3f}".format(class_name, idx, score), (0, 10), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (255, 0, 0), thickness=2)
    return img


def adjust_bbox_coord(img, xmin, ymin, xmax, ymax, bg='white'):
    crop = img[ymin:ymax, xmin:xmax, :]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpening_kernel = np.array([[-1, -1, -1, -1, -1], [-1, 2, 2, 2, -1], [-1, 2, 8, 2, -1], [-1, 2, 2, 2, -1], [-1, -1, -1, -1, -1]]) / 8.0
    gray = cv2.filter2D(gray, -1, sharpening_kernel)

    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    if bg == 'white' or 'black':
        thr = opt.BBOX_BORDER_THRESH
        thresh = cv2.copyMakeBorder(thresh[thr:-thr, thr:-thr], thr,
                                thr, thr, thr, borderType=cv2.BORDER_CONSTANT,
                                value=[0, 0, 0])
    else:
        thresh = cv2.copyMakeBorder(thresh[0:-8, 0:-8], 0,8,0,8, borderType=cv2.BORDER_CONSTANT)

    return thresh, crop


def find_max_contour(contours=None):
    max_idx = 0
    max_area = 0
    for x, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if max_area < area:
            max_area = area
            max_idx = x

    return max_idx, max_area


def select_background_color(bg=None):
    if bg == 'white':
        color = (255, 255, 255)
    elif bg == 'black':
        color = (0, 0, 0)
    elif bg == 'classic':
        color = (200, 200, 200)
    else:
        color = (255, 255, 255)
    return color


def get_cnt_bubble(demo, image, class_name, dets, class_thresh, bg='white'):
    count = 0
    bubbles = []
    value = []
    bboxes = []
    scores = []
    dets_bubbles = []
    for i in range(np.minimum(10, dets.shape[0])):  # sort bounding box of speech bubble
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > class_thresh:
            scores.append(score)
            ymin = int(bbox[1])
            value.append(ymin)
            bboxes.append(bbox)
    order_k = sorted(range(len(value)), key=lambda k: value[k])
    try:
        for bub_order in order_k:
            count += 1
            bbox = bboxes[bub_order]
            xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            thresh, crop = adjust_bbox_coord(image, xmin, ymin, xmax, ymax, bg=bg)
            crop_tmp = crop.copy()
            _, cnts, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            max_idx, max_area = find_max_contour(contours=cnts)
            crop_tmp = draw_bubble_contour(crop_tmp, class_name, scores[bub_order], cnts, max_idx, count)
            demo[ymin:ymax, xmin:xmax, :] = crop_tmp[:, :, :]
            bubble = alpha_blend_bubble(crop, cnts, max_idx)
            bubbles.append(bubble)
            dets_bubbles.append([xmin, ymin, xmax, ymax])
            color = select_background_color(bg=bg)

            #  remove speech bubble from image by filling background color
            cv2.drawContours(crop, [cnts[max_idx]], 0, color, -1)
            cv2.drawContours(crop, [cnts[max_idx]], 0, color, 10)

    except Exception as ex: print(ex)
    return demo, image, bubbles, dets_bubbles
