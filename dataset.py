import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import cv2
import random

import file
import preprocess
import config
import rotate
import crop
import flip
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

    def train_data_transform(self, idx):
        #print(self.image_list)
        #print(self.gt_list)

        image = preprocess.loadImage(self.image_list.pop(0))
        gt, gt_len = preprocess.loadText(self.gt_list.pop(0))

        select = random.randint(0, 3)
        select = 2

        if select == 0 and config.data_augmentation_rotate:
            image, gt = rotate.rotate(image, gt, gt_len)

        elif select == 1 and config.data_augmentation_crop:
            image, gt = crop.crop(image, gt, gt_len)

        elif select == 2 and config.data_augmentation_flip:
            image, gt = flip.flip(image, gt)
        else:
            #original
            pass
        time.sleep(3)


        x = preprocess.normalizeMeanVariance(image)

        #resize

        #HCW -> CHW
        x = torch.from_numpy(x).permute(2, 0, 1)

        #x = Variable(x.unsqueeze(0))

        return x, gt




