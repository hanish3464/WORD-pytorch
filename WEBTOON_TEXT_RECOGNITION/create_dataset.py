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
import imgproc


def adjustImageSize(w, h, orig_w, orig_h):
    w, h = (orig_w - w) / 2, (orig_h - h) / 2
    return w, h


def removeNoise(img=None, arr=None):
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


def makeLetter(canvas=None, label=None, width=None, height=None, color=None, font=None):
    canvas.text((width, height), label, fill=(color), font=font)


def determineFontSize(font=None, size=None):
    font = ImageFont.truetype(font, size)
    return font


def determineCanvasSize(canvas=None, label=None, font=None):
    w, h = canvas.textsize(label, font=font)
    w, h = adjustImageSize(w, h, config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
    return w, h


def saltPepperNoiseGenerator(image):
    row, col = image.shape
    s_vs_p = 0.01
    amount = 0.05
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in (row, col)]

    out[coords] = 255
    dx = [1, -1, 0, 0]
    dy = [0, 0, 1, -1]
    coord_copy = coords.copy()

    for x, y in zip(dx, dy):
        coord_copy[0] += x
        coord_copy[1] += y
        out[coord_copy] = 255

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in (row, col)]
    out[coords] = 0
    coord_copy = coords.copy()
    for j, k in zip(dx, dy):
        coord_copy[0] += j
        coord_copy[1] += k
        out[coord_copy] = 0

    return out


def chunkNoiseGenerator(copy):
    copy = np.array(copy)
    chunk = np.array([[0., 255., 255., 255., 0.],
                      [255., 255., 255., 255., 255.],
                      [255., 255., 255., 255., 255.],
                      [255., 255., 255., 255., 255.],
                      [0., 255., 255., 255., 0.]])

    coord_x = random.randint(int(copy.shape[0] *(2/3)), copy.shape[0] - chunk.shape[0])
    coord_y = random.randint(0, copy.shape[1] - chunk.shape[1])
    copy[coord_x:coord_x + chunk.shape[0], coord_y:coord_y + chunk.shape[1]] = chunk[:]
    return copy


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
            sys.stdout.write(
                'TRAINING IMAGE GENERATION: ({}/{}) \r'.format(cnt,
                                                               config.NUM_CLASSES * len(fonts) * config.MORPH_NUM * config.NOISE_GEN_NUM))
            sys.stdout.flush()

        for f in fonts:

            for v in range(config.MORPH_NUM):

                image, drawing = makeCanvas(width=config.IMAGE_WIDTH, height=config.IMAGE_HEIGHT,
                                            color=config.BACKGROUND)
                font_type = determineFontSize(font=f, size=config.FONT_SIZE)
                w, h = determineCanvasSize(canvas=drawing, label=character, font=font_type)
                makeLetter(canvas=drawing, label=character, width=w, height=h, color=config.FONT_COLOR, font=font_type)

                morph_templete = np.array(image.copy())
                kernel = np.ones((2, 2), np.uint8)

                if v == 1: morph_templete = cv2.erode(morph_templete, kernel, iterations=1)
                else: morph_templete = cv2.dilate(morph_templete, kernel, iterations=1)

                for x in range(config.NOISE_GEN_NUM):
                    copy = morph_templete.copy()
                    cnt += 1

                    if x == 0: copy = saltPepperNoiseGenerator(copy)
                    elif x == 1: copy = chunkNoiseGenerator(copy)
                    else: pass #origin data

                    copy = Image.fromarray(np.array(copy))
                    file_utils.saveImage(dir=IMAGE_PATH, img=copy, index=cnt)
                    file_utils.saveCSV(dir=IMAGE_PATH, dst=labels_csv, index=cnt, label=character, num=k)

    #added custom training data difficult to classify

    tranfer_img_list, _, _, _ = file_utils.get_files(config.TRANFSER_TRAIN_IMAGE_PATH)
    label_mapper = file_utils.makeLabelMapper(config.LABEL_PATH)
    test_txt = []; test_num = []

    print("[CUSTOM HANGUL DIFFICULT DATASET GENERATION : {}]".format(len(tranfer_img_list) * 3))

    for txt in config.TRANSFER_CASE:
        test_num.append(label_mapper[0].tolist().index(txt))
        test_txt.append(txt)

    for idx, in_path in enumerate(tranfer_img_list):
        k, character = test_num[idx], test_txt[idx]
        img = imgproc.loadImage(in_path)
        img = imgproc.cvtColorGray(img)
        for x in range(config.NOISE_GEN_NUM):
            copy = img.copy()
            cnt += 1

            if x == 0: copy = saltPepperNoiseGenerator(copy)
            elif x == 1: copy = chunkNoiseGenerator(copy)
            else: pass  # origin data

            copy = Image.fromarray(np.array(copy))
            file_utils.saveImage(dir=IMAGE_PATH, img=copy, index=cnt)
            file_utils.saveCSV(dir=IMAGE_PATH, dst=labels_csv, index=cnt, label=character, num=k)

    labels_csv.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='create datasets')
    parser.add_argument('--train', action='store_true', help='train data generation')
    parser.add_argument('--test', action='store_true', help='test data generation')
    args = parser.parse_args()

    createDataset(args)
