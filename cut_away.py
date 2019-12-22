import file_utils
from PIL import Image
import cv2
import os
import shutil

SIZE_RATIO = 2.0
PIXEL_THRESHOLD = 500
MIN_SIZE= 100
input = './images/'
result = './result/'
ratio = 0

if os.path.isdir(result): shutil.rmtree(result)
if not os.path.isdir(result):
    os.mkdir(result)

img_list, name_list = file_utils.listFiles(input)

def search_pixels(img, h_criteria):
    h, w, _ = img.shape
    arr = []
    arr.append(h_criteria)
    mag = 1
    while arr:
        h_c = arr.pop(0)
        if h_c < 0 or h_c >= h:
            print('[error]Image out of index')
            return 'err'
        if h_c < h_criteria - PIXEL_THRESHOLD or h_c >= h_criteria + PIXEL_THRESHOLD:
            print('It takes too long times. so, cut down image as initial criteria')
            return h_criteria

        if (img[h_c] == 0).all() or (img[h_c] == 255).all(): return h_c
        arr.append(h_criteria + mag)
        arr.append(h_criteria - mag)
        mag += 1


def down_size_image(img, width=None):
    global ratio
    h, w, _ = img.shape
    ratio_h = int((width * h) // w)
    ratio = w / width
    img = cv2.resize(img, (width, ratio_h), interpolation=cv2.INTER_CUBIC)
    return img


def cut_away_image(origin=None, copy=None, criteria=None, name=None, index=None):
    global ratio
    h, _, _ = copy.shape
    if h <= criteria:
        piece = origin[0:int(h * ratio), :, ::-1]
        if h >= MIN_SIZE: cv2.imwrite(result + name + '-' + str(index) + '.jpg', piece)
        return

    n_h = search_pixels(copy, criteria)
    piece = origin[0:int(n_h * ratio), :, ::-1]
    cv2.imwrite(result + name + '-' + str(index) + '.jpg', piece)

    cut_away_image(origin=origin[int(n_h * ratio):, :, :], copy=copy[n_h:, :, :], criteria=criteria, name=name,
              index=index + 1)


for k, in_path in enumerate(img_list):
    img = file_utils.loadImage(in_path)
    copy = file_utils.cpImage(img=img)
    copy = down_size_image(copy, width=500)  # Resize image for searching pixels to decrease times.
    height, width, _ = img.shape
    copy_height, copy_width, _ = copy.shape
    height_criteria = int(copy_width * SIZE_RATIO)
    print('test images : {}/{} : {}'.format(k+1, len(img_list), height_criteria), end= '\r')
    cut_away_image(origin=img, copy=copy, criteria=height_criteria, name=name_list[k], index=0)
