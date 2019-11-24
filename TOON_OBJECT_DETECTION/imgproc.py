import cv2
import numpy as np
from skimage import io
from model.utils.blob import im_list_to_blob
import config

# -*- coding: utf-8 -*-

def loadImage(img_file):
    img = io.imread(img_file)

    if img.shape[0] == 2:
        img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)
    return img

def adjustImageRatio(img):
    h, w, _ = img.shape
    if w // h >= 8:
        img = cv2.resize(img, (w, 2 * h), interpolation=cv2.INTER_LINEAR)
    elif h // w >= 8:
        img = cv2.resize(img, (w * 2, h), interpolation=cv2.INTER_LINEAR)
    return img

def createImageBorder(img, img_size=1024, color=[255,255,255]):
    h, w, _ = img.shape
    l = r = (img_size // 2 - w) // 2
    t = b = (img_size // 2 - h) // 2
    constant = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=color)
    return constant

def adjustImageBorder(img, img_size=128, color=[255,255,255]):
    h, w, _ = img.shape
    if h > w:
        l = r = (h-w)//2
        t = b = 0
    elif w > h:
        l = r = 0
        t = b = (w-h)//2
    else:
        t = b = l = r = 0
    constant = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=color)
    constant = cv2.copyMakeBorder(constant, h//4, h//4, w//4, w//4, cv2.BORDER_CONSTANT, value=color)
    img = cv2.resize(constant, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    return img

def cropBBoxOnImage(img, charBBox):
    x_min, y_min = charBBox[0]
    x_max, y_max = charBBox[2]
    x_min, y_min, x_max, y_max = int(round(x_min)), int(round(y_min)), int(round(x_max)), int(round(y_max))
    img = img[y_min:y_max, x_min:x_max, :]
    return img

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    img = in_img.copy().astype(np.float32)
    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img

def resize_aspect_ratio(img, user_defined_size, interpolation, mag_ratio=1):
    height, width, channel = img.shape

    mag_org_size = mag_ratio * max(height, width)

    if mag_org_size > user_defined_size: #
        mag_org_size = user_defined_size

    ratio = mag_org_size / max(height, width)
    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation = interpolation)

    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)

    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc


    target_h, target_w = target_h32, target_w32
    size_heatmap = (int(target_w/2), int(target_h/2))
    return resized, ratio, size_heatmap

def getImageBlob(im):
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= config.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in config.TEST_SCALES:
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > config.TEST_MAX_SIZE:
            im_scale = float(config.TEST_MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)