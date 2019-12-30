"""dataset.py"""
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import file_utils
import imgproc
from augmentation import Data_Augmentation
from gaussian import GenerateGaussian
import opt


class root_dataset(Dataset):

    def __init__(self, images_path, labels_path, image_size):
        self.image_size = image_size
        self.image_list, _, _, _ = file_utils.get_files(images_path)
        _, _, self.label_list, _ = file_utils.get_files(labels_path)

        self.gaussian_generator = GenerateGaussian(1024, opt.gaussian_region, opt.gaussian_affinity)

    def __len__(self):
        return len(self.image_list)

    def load_image_and_gt(self, idx):
        image = imgproc.loadImage(self.image_list[idx])
        char_bboxes = file_utils.loadJson(self.label_list[idx])

        return image, char_bboxes

    def train_data_resize(self, region_score_GT, affinity_score_GT):
        region_score_GT = cv2.resize(region_score_GT, (self.image_size // 2 , self.image_size // 2))
        affinity_score_GT = cv2.resize(affinity_score_GT, (self.image_size // 2, self.image_size // 2))
        return region_score_GT, affinity_score_GT

    def train_data_transform_webtoon(self, idx):

        ''' Prepare the data for training '''

        image, char_bboxes = self.load_image_and_gt(idx)
        region_score_GT = self.gaussian_generator.region(image, char_bboxes)
        affinity_score_GT = self.gaussian_generator.affinity(image, char_bboxes)
        region_score_GT, affinity_score_GT = self.train_data_resize(region_score_GT, affinity_score_GT)
        image = imgproc.normalizeMeanVariance(image)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        confidence = np.ones((region_score_GT.shape[0], region_score_GT.shape[1]), np.float32)

        ''' Augment the data for training '''
        data_augmentation = Data_Augmentation(image, region_score_GT, affinity_score_GT, confidence)
        image, region_score_GT, affinity_score_GT, confidence = data_augmentation.select_augmentation_method()

        ''' Convert the data for Model prediction '''
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        region_score_GT = torch.from_numpy(region_score_GT / 255).float()
        affinity_score_GT = torch.from_numpy(affinity_score_GT / 255).float()
        confidence = torch.from_numpy(confidence).float()

        return image, region_score_GT, affinity_score_GT, confidence


class webtoon_dataset(root_dataset):

    def __init__(self, images_path, labels_path, image_size):
        super(webtoon_dataset, self).__init__(images_path, labels_path, image_size)

    def __getitem__(self, idx):
        image , region_score_GT, affinity_score_GT, confidence = self.train_data_transform_webtoon(idx)

        return image, region_score_GT, affinity_score_GT, confidence

    def __len__(self):
        return super(webtoon_dataset, self).__len__()


