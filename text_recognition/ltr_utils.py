# -*- coding: utf-8 -*-

import codecs
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import cv2
import math
import opt


def flush(buffer, cnt):
    buffer = ''
    cnt = 0
    return buffer, cnt


def display_stdout(chars=None, space=None, img_name=None, MODE=None, save_to=None):
    # save result with regard to spacing words
    str_buffer = ''
    word = 0
    cnt = 0
    img_idx = 0
    bubble = 0
    with codecs.open(save_to, 'w') as res:
        for k, char in enumerate(chars):
            cnt += 1
            str_buffer += char
            if cnt == space[bubble][word]:
                str_buffer += ' '
                word += 1
                cnt = 0
            if word == len(space[bubble]):
                if MODE == 'stdout':
                    print(str_buffer)
                elif MODE == 'file':
                    res.write(str_buffer + '\n')
                else:
                    print(str_buffer)
                    res.write(str_buffer + '\n')

                bubble += 1
                word = 0
                str_buffer, cnt = flush(str_buffer, cnt)
            if len(chars) == img_idx + 1: return
            if img_name[img_idx] != img_name[img_idx + 1]:
                if MODE == 'stdout':
                    print('')
                elif MODE == 'file':
                    res.write('\n')
                else:
                    print('')
                    res.write('\n')
            img_idx += 1


def get_size(txt, font):
    test_img = Image.new('L', (1, 1))
    test_draw = ImageDraw.Draw(test_img)
    return test_draw.textsize(txt, font)


def parsing(target=None, criteria=None):
    target = target.strip(' ')
    arr = target.split(' ')
    parse_arr = []

    len_thresh = math.ceil(len(arr) / criteria)
    if len_thresh <= 0: len_thresh = 1
    if len(arr) >= len_thresh:
        for k in range(len(arr) // len_thresh):
            parse_arr.append(arr[k * len_thresh:(k + 1) * len_thresh])
        parse_arr.append(arr[(k + 1) * len_thresh:])
    if parse_arr == []:
        parse_arr.append(arr[:])

    return parse_arr


def gen_txt_to_image(load_from=None, warp_item=None):
    with codecs.open(load_from, 'r', encoding='utf-8') as f:
        labels = f.read().strip('\ufeff').splitlines()
    cnt = 0
    item_idx = 0
    for k, text in enumerate(labels):
        try:
            if text != '':  # remove useless blank
                bubble_image = warp_item[item_idx][0]
                line_boxes_coordinate = warp_item[item_idx][1]
                item_idx += 1
            else:
                continue

            if len(line_boxes_coordinate) == 0:  # It doesn't exist detected line text, but bubbles
                item_idx += 1
                continue

            # rearrange the number of text unit depending on the number of line boxes
            arr = parsing(target=text, criteria=len(line_boxes_coordinate))
            for x, slices in enumerate(arr):
                # generate image of text
                if slices == [''] or slices == []: continue
                slices = ' '.join(slices)
                fontsize = 24
                if slices == '': continue
                cnt += 1
                color_text = "black"
                color_background = "white"
                font = ImageFont.truetype('./text_recognition/font.ttf', fontsize)
                width, height = get_size(slices, font)
                img = Image.new('RGB', (width, height), color_background)
                d = ImageDraw.Draw(img)
                d.text((0, 0), slices, fill=color_text, font=font)
                img = np.array(img)
                eng_h, eng_w, _ = img.shape

                orig_xmin, orig_ymin = line_boxes_coordinate[x][0]
                orig_xmax, orig_ymax = line_boxes_coordinate[x][2]
                tmp_ymax, tmp_ymin = orig_ymax, orig_ymin
                tmp_xmax, tmp_xmin = orig_xmax, orig_xmin

                # generated English text is overwritten to line bounding box location of speech bubbles
                if eng_h <= orig_ymax - orig_ymin:

                    img = cv2.copyMakeBorder(np.array(img), ((orig_ymax - orig_ymin) - eng_h) // 2,
                                             ((orig_ymax - orig_ymin) - eng_h) // 2, 0, 0, cv2.BORDER_CONSTANT,
                                             value=[255, 255, 255])

                elif eng_h > orig_ymax - orig_ymin:
                    h_margin = eng_h - (orig_ymax - orig_ymin)
                    orig_ymin -= h_margin // 2
                    orig_ymax += h_margin // 2
                    if orig_ymin < 0:
                        move_y = abs(orig_ymin)
                        orig_ymin = 0
                        orig_ymax += move_y
                else: pass

                if eng_w <= orig_xmax - orig_xmin:  # copyMakeBorder
                    img = cv2.copyMakeBorder(np.array(img), 0, 0, ((orig_xmax - orig_xmin) - eng_w) // 2,
                                             ((orig_xmax - orig_xmin) - eng_w) // 2, cv2.BORDER_CONSTANT,
                                             value=[255, 255, 255])

                elif eng_w > orig_xmax - orig_xmin:
                    w_margin = eng_w - (orig_xmax - orig_xmin)
                    orig_xmin -= w_margin // 2
                    orig_xmax += w_margin // 2

                    if orig_xmin < 0:
                        move_x = abs(orig_xmin)
                        orig_xmin = 0
                        orig_xmax += move_x
                else: pass

                if bubble_image.shape[1] - orig_xmax >= opt.WARP_SPACE_THRESHOLD:
                    img = cv2.resize(img, (orig_xmax - orig_xmin, orig_ymax - orig_ymin), interpolation=cv2.INTER_CUBIC)
                    bubble_image[orig_ymin:orig_ymax, orig_xmin:orig_xmax, :] = img[:]
                else:
                    img = cv2.resize(img, (tmp_xmax - tmp_xmin, tmp_ymax - tmp_ymin), interpolation=cv2.INTER_CUBIC)
                    bubble_image[tmp_ymin:tmp_ymax, tmp_xmin:tmp_xmax, :] = img[:]

            # If there remains leavings of line bounding box, delete the boxes by filling the white color.
            leavings = line_boxes_coordinate[cnt:len(line_boxes_coordinate)]
            for leaving in leavings:
                xmin, ymin = leaving[0]
                xmax, ymax = leaving[2]
                cv2.rectangle(bubble_image, (xmin, ymin), (xmax, ymax), (255, 255, 255), -1)
            cnt = 0

        except Exception as ex:
            print('[error Info]: {} / ltr_utils.py'.format(type(ex)))