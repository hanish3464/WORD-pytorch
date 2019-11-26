import pandas as pd
import numpy as np
import imgproc
import torch
import file_utils
import cv2

class hangul_dataset(object):

    def __init__(self, csv_path, label_path, train_size):
        self.data = pd.read_csv(csv_path)
        self.size = train_size
        self.images = self.data.iloc[:, 0]
        self.labels = np.asarray(self.data.iloc[:, 1])
        self.labels_num = np.asarray(self.data.iloc[:, 2])
        self.allLabels = file_utils.loadText(label_path)
        self.labelOneHotVector = torch.zeros([len(self.allLabels)], dtype=torch.long)

    def __getitem__(self, index):
        return self.train_data_transform(index)

    def __len__(self):
        return len(self.data.index)

    def train_data_transform(self, index):
        image = imgproc.loadImage(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(image).float().unsqueeze(2).permute(2, 0, 1)
        label = self.labels[index]
        label_num = self.labels_num[index]

        return image, label_num
