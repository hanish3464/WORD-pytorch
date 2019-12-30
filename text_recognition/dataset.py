import pandas as pd
import numpy as np
import torch
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import random
import opt
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from PIL import Image, ImageFilter
import file_utils, imgproc

class Hangul_Dataset(object):

    def __init__(self, csv_path=None, label_path=None, image_size=None, train=None, blur=None, distort=None):
        self.data = pd.read_csv(csv_path, error_bad_lines=False)
        self.size = image_size
        self.images = self.data.iloc[:, 0]
        self.labels = np.asarray(self.data.iloc[:, 1])
        self.labels_num = np.asarray(self.data.iloc[:, 2])
        self.allLabels = file_utils.loadText(label_path)
        self.labelOneHotVector = torch.zeros([len(self.allLabels)], dtype=torch.long)
        self.FLAG = train
        self.blur = blur
        self.distort = distort

    def __getitem__(self, index):
        if self.FLAG: return self.train_data_transform(index)
        else: return self.test_data_transform(index)

    def __len__(self):
        return len(self.data.index)

    def test_data_transform(self, index):
        image = imgproc.loadImage(self.images[index])
        image = imgproc.cvtColorGray(image)
        image = imgproc.tranformToTensor(img=image, size=self.size)

        label = self.labels[index]
        label_num = self.labels_num[index]

        return image, label_num

    def train_data_transform(self, index):
        image = imgproc.loadImage(self.images[index])
        image = imgproc.cvtColorGray(image)

        # Data Augmentation Method - elastic distortion, image blur

        if self.distort:
            if random.randint(0, 1):
                image = self.distort_image(image)

        if self.blur:
            if random.randint(0, 1):
                blur_extent = 1
                image = self.blur_image(image, blur_extent)

        image = imgproc.tranformToTensor(img=image, size=self.size)

        label = self.labels[index]
        label_num = self.labels_num[index]

        return image, label_num

    def distort_image(self, img):

        alpha = random.randint(opt.ALPHA_MIN, opt.ALPHA_MAX)
        sigma = random.randint(opt.SIGMA_MIN, opt.SIGMA_MAX)

        random_state = np.random.RandomState(None)
        shape = img.shape

        dx = gaussian_filter(
            (random_state.rand(*shape) * 2 - 1),
            sigma, mode="constant"
        ) * alpha
        dy = gaussian_filter(
            (random_state.rand(*shape) * 2 - 1),
            sigma, mode="constant"
        ) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        distorted_arr = map_coordinates(img, indices, order=1).reshape(shape)
        #image = Image.fromarray(distorted_arr)
        return distorted_arr

    def blur_image(self, img, extent):
        img = Image.fromarray(img)
        img = img.filter(ImageFilter.GaussianBlur(radius=extent))
        return np.array(img)
