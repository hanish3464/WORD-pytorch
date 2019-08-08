"""  
preprocess.py
"""

# -*- coding: utf-8 -*-
import numpy as np
from skimage import io
import cv2
import codecs
import debug
from PIL import Image
def loadImage(img_file): #HCW order ->image channel adjuestment
    img = io.imread(img_file)           # RGB order

    if img.shape[0] == 2:
        img = img[0] #Height is 2, height will be img itself. but height is not 2
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) #Channel is 2, fix 3
    if img.shape[2] == 4:   img = img[:,:,:3] #Channel is 4, fix 3
    img = np.array(img) #convert list to numpy

    return img

def loadText(txt_file):
    with codecs.open(txt_file, encoding='utf-8_sig') as file:
        coordinate = file.readlines()

        for line in coordinate:
            tmp = line.split(',')
            arr_coordinate = [int(n) for n in tmp]
            coordinate = np.array(arr_coordinate).reshape([1,4,2])

    return coordinate

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order #Z-score conversion. image pixel value is changed to predict easily.
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)

    return img

def denormalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def resize_aspect_ratio(img, user_defined_size, interpolation, mag_ratio=1):
    height, width, channel = img.shape

    # magnify image size
    mag_org_size = mag_ratio * max(height, width) #usually height is selected as max value

    # set original image size
    if mag_org_size > user_defined_size: #
        mag_org_size = user_defined_size

    ratio = mag_org_size / max(height, width)
    target_h, target_w = int(height * ratio), int(width * ratio) #image size magnified
    proc = cv2.resize(img, (target_w, target_h), interpolation = interpolation)

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0: #after diveded by 32, add remains to target_h,w
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)

    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc #external black padding create
    """debug Image external black padding addition"""
    debugTmpImg = np.zeros((target_h32, target_w32, channel), dtype=np.uint8)
    debugTmpImg[0:target_h, 0:target_w, :] = proc

    target_h, target_w = target_h32, target_w32
    size_heatmap = (int(target_w/2), int(target_h/2)) #heatmap size is half of target
    return resized, ratio, size_heatmap

def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img
