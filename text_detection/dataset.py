"""train.py"""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import file_utils
import imgproc
import config
from augmentation import Data_augmentation
from gaussian import GenerateGaussian


class root_dataset(Dataset):

    def __init__(self, images_path, char_gt_path, word_gt_path, image_size):
        self.image_size = image_size
        self.image_list, _, _ = file_utils.get_files(images_path)
        _, _, self.char_gt_list = file_utils.get_files(char_gt_path)

        self.gaussian_generator = GenerateGaussian(1024, config.gaussian_region, config.gaussian_affinity)

    def __len__(self):
        return len(self.image_list)

    def load_image_and_gt(self, idx):
        image = imgproc.loadImage(self.image_list[idx])
        name = imgproc.loadName(self.image_list[idx])
        item = {'image': image, 'name': name}
        charBBox, charBBox_len= imgproc.loadText(self.char_gt_list[idx])
        wordBBox, wordBBox_len = imgproc.loadText(self.word_gt_list[idx])
        return item, charBBox, wordBBox, charBBox_len, wordBBox_len

    def train_data_resize(self, region_score_GT, affinity_score_GT):
        region_score_GT = cv2.resize(region_score_GT, (self.image_size // 2 , self.image_size // 2))
        affinity_score_GT = cv2.resize(affinity_score_GT, (self.image_size // 2, self.image_size // 2))
        return region_score_GT, affinity_score_GT

    def train_data_transform_webtoon(self, idx):

        ''' Prepare the data for training '''

        item, charBBox, wordBBox, charBBox_len, wordBBox_len = self.load_image_and_gt(idx)
        region_score_GT = self.gaussian_generator.region(item, charBBox, wordBBox, charBBox_len, wordBBox_len)
        affinity_score_GT = self.gaussian_generator.affinity(item, charBBox, wordBBox, charBBox_len, wordBBox_len)
        region_score_GT, affinity_score_GT = self.train_data_resize(region_score_GT, affinity_score_GT)
        image = cv2.cvtColor(item['image'], cv2.COLOR_BGR2RGB)
        confidence = np.ones((region_score_GT.shape[0], region_score_GT.shape[1]), np.float32)

        ''' Augment the data for training '''
        data_augmentation = Data_augmentation(image, region_score_GT, affinity_score_GT, confidence)
        image, region_score_GT, affinity_score_GT, confidence = data_augmentation.select_augmentation_method()

        ''' Convert the data for Model prediction '''
        image = imgproc.normalizeMeanVariance(image)
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        region_score_GT = torch.from_numpy(region_score_GT / 255).float()
        affinity_score_GT = torch.from_numpy(affinity_score_GT / 255).float()
        confidence = torch.from_numpy(confidence).float()

        return image, region_score_GT, affinity_score_GT, confidence


class webtoon_dataset(root_dataset):

    def __init__(self, images_path, char_gt_path, word_gt_path, image_size):
        super(webtoon_dataset, self).__init__(images_path, char_gt_path, word_gt_path, image_size)

    def __getitem__(self, idx):
        image , region_score_GT, affinity_score_GT, confidence = self.train_data_transform_webtoon(idx)

        return image, region_score_GT, affinity_score_GT, confidence

    def __len__(self):
        return super(webtoon_dataset, self).__len__()




