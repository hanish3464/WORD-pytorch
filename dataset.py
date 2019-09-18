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
from gaussian import GenerateGaussian
import time

class root_dataset(Dataset):

    def __init__(self, images_path, char_gt_path, word_gt_path, image_size):
        self.image_size = image_size
        self.image_list, _, _ = file.get_files(images_path)
        _, _, self.char_gt_list = file.get_files(char_gt_path)
        _, _, self.word_gt_list = file.get_files(word_gt_path)

        self.gaussian_generator = GenerateGaussian(config.gaussian_sigma, config.gaussian_spread)
        print('success gaussian_generator')
    def __len__(self):
        return len(self.image_list)

    def load_image_and_gt(self, idx):
        image = preprocess.loadImage(self.image_list[idx])
        name = preprocess.loadName(self.image_list[idx])
        item = {'image': image, 'name': name}
        charBBox, charBBox_len= preprocess.loadText(self.char_gt_list[idx])
        wordBBox, wordBBox_len = preprocess.loadText(self.word_gt_list[idx])
        return item, charBBox, wordBBox, charBBox_len, wordBBox_len

    def train_data_transform(self, idx):
        item, charBBox, wordBBox, charBBox_len, wordBBox_len = self.load_image_and_gt(idx)
        region_score_GT = self.gaussian_generator.region(item, charBBox, wordBBox, charBBox_len, wordBBox_len)
        affinity_score_GT = self.gaussian_generator.affinity(item, charBBox, wordBBox, charBBox_len, wordBBox_len)
        render_img = region_score_GT.copy()
        render_img = np.hstack((render_img, affinity_score_GT))
        cv2.imwrite('./train/mask/mask_'+str(idx)+'.jpg', render_img)


        image = preprocess.normalizeMeanVariance(item['image'])
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        region_score_GT = torch.from_numpy(region_score_GT / 255).float()
        affinity_score_GT = torch.from_numpy(affinity_score_GT / 255).float()

        return image.double(), region_score_GT.double(), affinity_score_GT.double()

class webtoon_dataset(root_dataset):

    def __init__(self, images_path, char_gt_path, word_gt_path, image_size):
        super(webtoon_dataset, self).__init__(images_path, char_gt_path, word_gt_path, image_size)

    def __getitem__(self, idx):
        image , region_score_GT, affinity_score_GT  = self.train_data_transform(idx)

        return {'image': image, 'region_score_GT': region_score_GT, 'affinity_score_GT': affinity_score_GT}

    def __len__(self):
        return super(webtoon_dataset, self).__len__()







