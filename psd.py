# -*- coding: utf-8 -*-
# Add Library
from psd_tools import PSDImage
from psd_tools.constants import Resource
from psd_tools import compose
from psd_tools.psd.layer_and_mask import MaskData
from psd_tools.api import effects
from PIL import Image
import os
import math
import cv2
import numpy as np

items=list()

def list_files(in_path):
    psd_files = list()
    psd_names = list()
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.psd':
                psd_files.append(os.path.join(dirpath, file))
                psd_names.append(filename)
    psd_files.sort()
    psd_names.sort()
    return psd_files, psd_names

def combination(arr, r):
    global items
    arr = sorted(arr)
    used = [0 for _ in range(len(arr))]
    def generate(chosen):
        if len(chosen) == r:
            items.append(chosen)
            return
        start = arr.index(chosen[-1]) + 1 if chosen else 0
        for nxt in range(start, len(arr)):
            if used[nxt] == 0 and (nxt == 0 or arr[nxt-1] != arr[nxt] or used[nxt-1]):
                chosen.append(arr[nxt])
                used[nxt] = 1
                generate(chosen)
                chosen.pop()
                used[nxt] = 0
    generate([])

def writeText(pixels, w, h, path, idx):
    xmin, ymin, xmax, ymax, xflag, yflag = 0, 0, 0, 0, True, True

    for i, pixel in enumerate(pixels):
        if pixel[3] != 255: continue
        else:
            y = int(i / w)
            x = (i - 1) % w

            if yflag:
                ymin = y
                yflag = False
            if xflag:
                xmin = x
                xflag = False
            elif xmin > x:
                xmin = x
            if ymax < y:
                ymax = math.ceil(y)
            if xmax < x:
                xmax = math.ceil(x)

    f = open(path + str(idx) + ".txt", "w")
    print("xmin : {} ymin : {}".format(xmin, ymin))
    print("xmax : {} ymax : {}".format(xmax, ymax))
    f.write("{},{},{},{}".format(int(xmin), int(ymin), int(xmax), int(ymax)))
    f.close()


def main(psd_path, png_path, png_text_path):
    cnt_b, cnt_t = 0,0
    text, bubble, only_text = list(), list(), list()
    psd_files_list, psd_name_list = list_files(psd_path)
    #psd_files_list = psd_files_list[13:]
    print(psd_files_list)
    if not os.path.isdir(png_path):
        os.makedirs(png_path)
    tmp = 9
    for idx, psd_file in enumerate(psd_files_list):
        psd = PSDImage.open(psd_file)
        try:
            if len(psd) == 0: continue
            image_size = psd[0].bbox
            print(image_size)
            for layer in psd:
                if layer.name == u'말칸' or layer.name == u'외침':
                    layer.visible = True
                    #bubble.append(layer.compose(image_size))
                    cnt_b += 1

                if layer.name == u'대사' or layer.kind == u'type' or layer.name[:1] == 'w':
                    layer.visible = True
                    #layer.compose(bbox=image_size).save('{}{}.png'.format(png_text_path+ 'tmp', tmp))
                    tmp_text = layer.compose(bbox=image_size)
                    if tmp_text is None: continue
                    cnt_t += 1

            if cnt_t > 0 and cnt_b > 0:
                if cnt_t == cnt_b:
                    tmp_text = np.asarray(tmp_text)
                    tmp_text = cv2.resize(tmp_text, (1024, 1024), interpolation=cv2.INTER_LINEAR)
                    tmp_text = Image.fromarray(tmp_text)
                    pixelData = list(tmp_text.getdata())
                    w, h = tmp_text.size
                    writeText(pixelData, w, h, png_text_path + 'tmp', tmp)

                    psd_tmp = psd.compose(psd)
                    psd_tmp = np.asarray(psd_tmp)
                    psd_tmp = cv2.resize(psd_tmp, (1024,1024), interpolation = cv2.INTER_LINEAR)
                    psd_tmp = Image.fromarray(psd_tmp)
                    psd_tmp.save('{}{}.png'.format(png_path + 'tmp', tmp))
                    tmp += 1
                    print("PSD:({}/{})".format(idx + 1, len(psd_files_list)))

        except Exception as ex:
            print(ex)


        #bubble, text = [], []
        cnt_b, cnt_t = 0, 0


if __name__ == '__main__':
    psd_path = '/home/hanish/psd/psd/'
    png_path = '/home/hanish/psd/png/original_img/'
    png_text_path = '/home/hanish/psd/png/only_text/'
    main(psd_path, png_path, png_text_path)

 # copys = PSDImage.open(psd_file)
            # if len(psd) == 0 or len(copys) == 0: continue
            # for copy in copys:
            #     if copy.name == u'말칸' or copy.name == u'외침':
            #         copy.visible = True
            #     elif copy.name == u'대사' or copy.kind == u'type' or copy.name[:1] == 'w':
            #         copy.visible = True
            # if cnt_t > cnt_b:
            #     print('cnt_t > cnt_b')
            #     psd.compose(psd)
            #     for bub in bubble:
            #         psd = Image.alpha_composite(psd, bub)
            #     psd2 = psd
            #     global items
            #     for k in range(len(items)):
            #         for txt in items[k]:
            #             psd2 = Image.alpha_composite(psd2, txt)
            #         psd2.save('{}{}_{}.png'.format(png_path + 'tmp', tmp, k))
            #         psd2 = psd
            #     num = 0
            #     while True:
            #         head = only_text[0]
            #         head.save('{}{}_{}.png'.format(png_text_path + 'tmp', tmp, num))
            #         only_text = only_text[1:]
            #         num +=1
            #         if not only_text: break
