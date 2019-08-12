from PIL import Image
import numpy as np
import config
import random

def rotate_both_img_and_gt(gt, data = None):
    gt = gt.copy()

    if data is not None:
        img = Image.fromarray(np.uint8(data))
    origin_w, origin_h = img.size

    #rotate image
    rotate_rand = random.randint(0,3)
    angle = rotate_rand * 90
    img = img.rotate(30, expand = True)
    img = np.array(img)

    #rotate ground truth
    rotate_gt = np.array(gt).reshape([-1,4,2])
    print(origin_w, origin_h)
    #for i in range(angle // 90):
        #rotate_gt[:, :, 0] = origin_w - 1 - rotate_gt[:, :, 0]
        #rotate_gt[:, :, [0, 1]] = rotate_gt[:, :, [1, 0]]
        #origin_h, origin_w = origin_w, origin_h
    new_label = rotate_gt.reshape([-1, 8]).astype(int)
    rotate_gt = rotate_gt.tolist()

    return rotate_gt, img, angle

def rotation(img, gt):
    rotate_rand = random.random() if config.data_augmentation_rotate else 0
    if rotate_rand > 0.0:
        gt, img, angle = rotate_both_img_and_gt(gt, data= img)
    return img, gt


