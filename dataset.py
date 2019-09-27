import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import cv2
import random
import numpy as np
import re
import itertools
import scipy.io as scio
import file
import preprocess
import config
from augmentation import Data_augmentation
from gaussian import GenerateGaussian
import os

def random_scale(img, bboxes, min_size):
    h, w = img.shape[0:2]
    scale = 1.0
    if max(h, w) > 1280:
        scale = 1280.0 / max(h, w)
    random_scale = np.array([1.0, 2.0, 3.0])
    scale1 = np.random.choice(random_scale)
    if min(h, w) * scale * scale1 <= min_size:
        scale = (min_size + 10) * 1.0 / min(h, w)
    else:
        scale = scale * scale1
    bboxes *= scale
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img

class root_dataset(Dataset):

    def __init__(self, images_path, char_gt_path, word_gt_path, image_size):
        self.image_size = image_size
        self.image_list, _, _ = file.get_files(images_path)
        _, _, self.char_gt_list = file.get_files(char_gt_path)
        _, _, self.word_gt_list = file.get_files(word_gt_path)

        self.gaussian_generator = GenerateGaussian(1024, config.gaussian_region, config.gaussian_affinity)
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
        image = preprocess.normalizeMeanVariance(image)
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        region_score_GT = torch.from_numpy(region_score_GT / 255).float()
        affinity_score_GT = torch.from_numpy(affinity_score_GT / 255).float()
        confidence = torch.from_numpy(confidence).float()


        return image, region_score_GT, affinity_score_GT, confidence

class webtoon_dataset(root_dataset):

    def __init__(self, images_path, char_gt_path, word_gt_path, image_size):
        super(webtoon_dataset, self).__init__(images_path, char_gt_path, word_gt_path, image_size)

    def __getitem__(self, idx):
        image , region_score_GT, affinity_score_GT, confidence  = self.train_data_transform_webtoon(idx)

        return image, region_score_GT, affinity_score_GT, confidence

    def __len__(self):
        return super(webtoon_dataset, self).__len__()

class synthText_dataset(root_dataset):

    def __init__(self, synthText_path, target_size=768):
        super(synthText_dataset, self).__init__(target_size)
        self.synthtext_folder = synthText_path
        gt = scio.loadmat(os.path.join(synthText_path, 'gt.mat'))
        self.charbox = gt['charBB'][0]
        self.image = gt['imnames'][0]
        self.imgtxt = gt['txt'][0]

    def __getitem__(self, index):
        return self.pull_item(index)

    def __len__(self):
        return len(self.imgtxt)

    def get_imagename(self, index):
        return self.image[index][0]

    def load_image_gt(self, index):

        img_path = os.path.join(self.synthtext_folder, self.image[index][0])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _charbox = self.charbox[index].transpose((2, 1, 0))
        image = random_scale(image, _charbox, self.target_size)

        words = [re.split('\n|\n |\n|', t.strip()) for t in self.imgtxt[index]]
        words = list(itertools.chain(*words))
        words = [t for t in words if len(t) > 0]
        character_bboxes = []
        total = 0
        confidences = []
        for i in range(len(words)):
            bboxes = _charbox[total:total + len(words[i])]
            assert (len(bboxes) == len(words[i]))
            total += len(words[i])
            bboxes = np.array(bboxes)
            character_bboxes.append(bboxes)
            confidences.append(1.0)

        return image, character_bboxes, words, np.ones((image.shape[0], image.shape[1]), np.float32), confidences



