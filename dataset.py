import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import cv2
import random
import numpy as np

import file
import preprocess
import config
import rotate
import crop
import flip
import resize
import time

class webtoon_text_detection_dataset(Dataset):

    def __init__(self, images_path, ground_truth_path):
        self.image_list, _, _ = file.get_files(images_path)
        #print(self.image_list)
        _, _, self.gt_list = file.get_files(ground_truth_path)

    def __getitem__(self, idx):
        image , gt = self.train_data_transform(idx)
        """We should generate gaussian heat map & character region parsing for training about returned gt"""
        return {'image': image, 'gt': gt}

    def __len__(self):
        return config.train_img_num

    @staticmethod
    def check_data_augmentation_method(image, gt):
        new_gt_temp = gt[:]
        while True:
            if not new_gt_temp:
                break
            new_gt_element = new_gt_temp.pop()
            poly = np.array(new_gt_element).astype(np.int32).reshape((-1)).reshape(-1, 2)
            cv2.polylines(image, [poly.reshape((-1, 1, 2))], True, color=(255, 0, 0), thickness=1)
            ptColor = (0, 255, 255)
        time.sleep(0.5)
        cv2.imwrite('/home/hanish/workspace/debug_image/debug_final.jpg', image)

    def train_data_transform(self, idx):
        print(idx)
        #print(self.image_list)
        #print(self.gt_list)
        print('load img')
        image = preprocess.loadImage(self.image_list[idx])
        gt, gt_len = preprocess.loadText(self.gt_list[idx])
        select = random.choice(['rotate', 'crop', 'flip', 'origin'])



        if select == 'rotate' and config.data_augmentation_rotate:
            image, gt = rotate.rotate(image, gt, gt_len)
            #print("rotate : " + str(image.shape))

        elif select == 'crop' and config.data_augmentation_crop:
            image, gt, gt_len = crop.crop(image, gt, gt_len)
            #print("crop : " + str(image.shape))

        elif select == 'flip' and config.data_augmentation_flip:
            image, gt = flip.flip(image, gt)
            #print("flip : " + str(image.shape))
        elif select == 'origin':
            #original
            pass

        # 512 x 512 image resize
        image, gt = resize.resize_gt(image, gt, gt_len)
        self.check_data_augmentation_method(image, gt)
        #print("resized : " + str(image.shape))

        x = preprocess.normalizeMeanVariance(image)

        #HCW -> CHW
        x = torch.from_numpy(x).permute(2, 0, 1)

        #x = Variable(x.unsqueeze(0))

        return x, gt




