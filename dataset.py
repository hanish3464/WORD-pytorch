import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import cv2
import file
import preprocess
import config
import rotate
import debug

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
        image = preprocess.loadImage(self.image_list.pop(0))
        gt = preprocess.loadText(self.gt_list.pop(0))
        #image, _, _= \
           # preprocess.resize_aspect_ratio(image, config.image_size, interpolation=cv2.INTER_LINEAR, mag_ratio=config.mag_ratio)

        rotated_img, rotated_gt = rotate.rotation(image, gt)
        print(rotated_gt)
        debug.printing(rotated_img)

        x = preprocess.normalizeMeanVariance(image)

        #resize

        #HCW -> CHW
        x = torch.from_numpy(x).permute(2, 0, 1)

        #x = Variable(x.unsqueeze(0))

        return x, gt




