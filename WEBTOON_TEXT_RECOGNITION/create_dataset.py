#! /usr/bin/env/ python
# -*- coding: utf-8 -*-

import glob
import codecs
import os
import random
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageFilter
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import config
import file_utils
import cv2
import sys
import argparse

def adjustImageSize(w, h, orig_w, orig_h):
    w, h = (orig_w - w) / 2, (orig_h - h) / 2
    return w, h
def removeNoise(img=None, arr = None):
    h, w = img.shape
    arr[0:5, 0:] = 0
    arr[h - 5:, 0:] = 0
    arr[0:, 0:5] = 0
    arr[0:, w - 5:] = 0
    return arr

def rotateImage(img=None, angle=None):
    image = np.array(img)
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_W = int((h * sin) + (w * cos))
    new_H = int((h * cos) + (w * sin))

    M[0, 2] += (new_W / 2) - center_x
    M[1, 2] += (new_H / 2) - center_y

    rotated_arr = cv2.warpAffine(image, M, (new_W, new_H), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    rotated_arr = cv2.resize(rotated_arr, (64, 64), interpolation=cv2.INTER_CUBIC)
    rotated_image = Image.fromarray(rotated_arr)
    return rotated_image

def blurImage(img=None, extent=None):
    img = img.filter(ImageFilter.GaussianBlur(radius=extent))
    return img

def distortImage(img=None):

    alpha = random.randint(config.ALPHA_MIN, config.ALPHA_MAX)
    sigma = random.randint(config.SIGMA_MIN, config.SIGMA_MAX)

    random_state = np.random.RandomState(None)
    shape = img.shape

    dx = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma, mode="constant"
    ) * alpha
    dy = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma, mode="constant"
    ) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    distorted_arr = map_coordinates(img, indices, order=1).reshape(shape)
    distorted_image = Image.fromarray(distorted_arr)

    return distorted_image

def makeCanvas(width=None, height=None, color=None):
    image = Image.new('RGB', (width, height), color=color)
    drawing = ImageDraw.Draw(image)
    return image, drawing

def makeLetter(canvas=None, label= None, width=None, height=None, color=None, font=None):
    canvas.text((width, height), label, fill=(color), font=font)

def determineFontSize(font=None, size=None):
    font = ImageFont.truetype(font, size)
    return font
def determineCanvasSize(canvas=None, label=None, font=None):
    w, h = canvas.textsize(label, font=font)
    w, h = adjustImageSize(w, h, config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
    return w, h

def createDataset(args):
    with codecs.open(config.LABEL_PATH, 'r', encoding='utf-8') as f:
        labels = f.read().strip('\ufeff').splitlines()
    if args.train:
        FONTS_PATH = config.TRAIN_FONTS_PATH
        CSV_PATH = config.TRAIN_CSV_PATH
        IMAGE_PATH = config.TRAIN_IMAGE_PATH
    elif args.test:
        FONTS_PATH = config.TEST_FONTS_PATH
        CSV_PATH = config.TEST_CSV_PATH
        IMAGE_PATH = config.TEST_IMAGE_PATH
    else:
        FONTS_PATH = config.TRAIN_FONTS_PATH
        CSV_PATH = config.TRAIN_CSV_PATH
        IMAGE_PATH = config.TRAIN_IMAGE_PATH


    fonts = glob.glob(os.path.join(FONTS_PATH, '*.ttf'))
    labels_csv = codecs.open(os.path.join(CSV_PATH), 'w', encoding='utf-8')
    print("[THE NUMBER OF FONTS : {}]".format(len(fonts)))
    cnt = 0
    prev_cnt = 0
    for k, character in enumerate(labels):

        if cnt - prev_cnt > 5000:
            prev_cnt = cnt
            sys.stdout.write('TRAINING IMAGE GENERATION: ({}/{}) \r'.format(cnt, 2421 * len(fonts)))
            sys.stdout.flush()

        for font in fonts:
            cnt += 1

            image, drawing = makeCanvas(width=config.IMAGE_WIDTH, height=config.IMAGE_HEIGHT, color=config.BACKGROUND)

            font = determineFontSize(font=font, size=config.FONT_SIZE)

            w, h = determineCanvasSize(canvas=drawing, label=character, font=font)
            makeLetter(canvas=drawing, label=character, width=w, height=h, color=config.FONT_COLOR, font=font)

            file_utils.saveImage(dir=IMAGE_PATH, img=image, index=cnt)
            file_utils.saveCSV(dir=IMAGE_PATH, dst=labels_csv, index=cnt, label=character, num=k)

            # for _ in range(config.DISTORTION_TIMES):
            #
            #     cnt += 1
            #
            #     arr = np.array(image)
            #     distorted_image = distortImage(img=arr)
            #     file_utils.saveImage(dir=config.TRAIN_IMAGE_PATH, img=distorted_image, index=cnt)
            #     file_utils.saveCSV(dir=config.TRAIN_IMAGE_PATH, dst=labels_csv, index=cnt, label=character)
            #
            #     cnt += 1
            #     blur_extent = random.randint(1, 4)
            #     blur_image = blurImage(img=distorted_image, extent=blur_extent)
            #     file_utils.saveImage(dir=config.TRAIN_IMAGE_PATH, img=blur_image, index=cnt)
            #     file_utils.saveCSV(dir=config.TRAIN_IMAGE_PATH, dst=labels_csv, index=cnt, label=character)
            #
            #     cnt += 1
            #     angle = random.choice([10*random.randint(33, 36), 10*random.randint(1, 3)])
            #     rotated_image = rotateImage(img=distorted_image, angle=angle)
            #     file_utils.saveImage(dir=config.TRAIN_IMAGE_PATH, img=rotated_image, index=cnt)
            #     file_utils.saveCSV(dir=config.TRAIN_IMAGE_PATH, dst=labels_csv, index=cnt, label=character)

    labels_csv.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='create datasets')
    parser.add_argument('--train', action='store_true', help='train data generation')
    parser.add_argument('--test', action='store_true', help='test data generation')
    args = parser.parse_args()

    createDataset(args)

