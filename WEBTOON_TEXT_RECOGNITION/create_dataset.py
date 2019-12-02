#! /usr/bin/env/ python
# -*- coding: utf-8 -*-

import glob
import codecs
import os
import random
import numpy as np
from PIL import Image, ImageFont, ImageDraw

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

def makeCanvas(width=None, height=None, color=None):
    image = Image.new('L', (width, height), color=color)
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
            sys.stdout.write('TRAINING IMAGE GENERATION: ({}/{}) \r'.format(cnt, config.NUM_CLASSES * len(fonts)))
            sys.stdout.flush()

        for f in fonts:

            #for _ in range(8):

            cnt += 1

                #w_rand = random.randint(0, 8)
                #h_rand = random.randint(0, 8)
                #config.IMAGE_WIDTH = 48 + (8 * w_rand)
                #config.IMAGE_HEIGHT = 48 + (8 * h_rand)

            image, drawing = makeCanvas(width=config.IMAGE_WIDTH, height=config.IMAGE_HEIGHT, color=config.BACKGROUND)
            font_type = determineFontSize(font=f, size=config.FONT_SIZE)
            w, h = determineCanvasSize(canvas=drawing, label=character, font=font_type)
            makeLetter(canvas=drawing, label=character, width=w, height=h, color=config.FONT_COLOR, font=font_type)

            file_utils.saveImage(dir=IMAGE_PATH, img=image, index=cnt)
            file_utils.saveCSV(dir=IMAGE_PATH, dst=labels_csv, index=cnt, label=character, num=k)

    labels_csv.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='create datasets')
    parser.add_argument('--train', action='store_true', help='train data generation')
    parser.add_argument('--test', action='store_true', help='test data generation')
    args = parser.parse_args()

    createDataset(args)

