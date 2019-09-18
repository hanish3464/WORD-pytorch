"""train.py"""

import torch
from torch import optim
from torch.utils.data import DataLoader
import os
import torch.backends.cudnn as cudnn
from wtd import WTD
import config
from loss import WTD_LOSS
from dataset import webtoon_dataset

import debug


import time
import preprocess
import postprocess
import numpy as np
import sys

def train_net(myNet, device, dataloader, optimizer, iteration):






def train():
    """there is under developing"""

    datasets = webtoon_dataset(config.train_images_path, config.train_gt_path, image_size = config.train_image_size)
    train_data_loader = DataLoader(datasets, batch_size = config.batch, shuffle = True, num_workers = config.num_of_gpu, drop_last = True, pin_memory = True)


    myNet = WTD()


    if config.cuda: #GPU
        device = torch.device("cuda:0")
        myNet = myNet.cuda()
    else: #CPU
        device = torch.device("cpu")


    optimizer = optim.Adam(myNet.parameters(), lr = config.lr, weight_decay = config.weight_decay)
    cudnn.benchmark = True
    criterion = WTD_LOSS()

    if not os.path.isdir(config.train_prediction_path):  os.mkdir(config.train_prediction_path)

    #myNet.train()


    iteration = 0

    for i in range(config.epoch):
        print('epoch :{} entered'.format(i))
        for i_batch, sample in enumerate(train_data_loader):
            images = sample['image'].to(device, dtype=torch.float)
            # gt_text= sample['gt_text'].to(device, dtype=torch.float)
            # gt_affinity= sample['gt_affinity'].to(device, dtype=torch.float)
            # gt_text = Variable(gt_text).cuda()
            # gt_affinity = Variable(gt_affinity).cuda()

            y, _ = myNet(images)

            score_text = y[0, :, :, 0].cpu().data.numpy()  # what is y dimension
            score_affinity = y[0, :, :, 1].cpu().data.numpy()

            '''loss function'''

            # optimizer.zero_grad()

            # loss = criterion(gt_text, gt_affinity, score_text, score_affinity)
            # loss.backward()
            # optimizer.step()

            if iteration % 5 is 0:  # exist bug
                debug.printing(score_text)
                debug.printing(score_affinity)

            iteration += 1
