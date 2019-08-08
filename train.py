"""train.py"""

import torch
from torch import optim
from torch.utils.data import DataLoader
import os

from wtd import WTD
import config
import file
import dataset

import debug


import time
import preprocess
import postprocess

def train_net(myNet, device, dataloader, optimizer, iteration):

    for i in range(config.epoch):
        print('epoch :{} entered'.format(i))
        for i_batch, sample in enumerate(dataloader):
            images = sample['image'].to(device)
            y, _ = myNet(images)

            #loss function

            #optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()

            score_text = y[0, :, :, 0].cpu().data.numpy()
            score_link = y[0, :, :, 1].cpu().data.numpy()

            if iteration % config.iterations is 0: #exist bug
                debug.printing(score_text)
                debug.printing(score_link)

            iteration += 1




def train():
    """there is under developing"""
    print(config.train_images_folder_path)
    datasets = dataset.webtoon_text_detection_dataset(config.train_images_folder_path, config.train_ground_truth_folder)
    dataloader = DataLoader(datasets, batch_size = config.batch_size, shuffle = True, num_workers = config.num_of_gpu)
    myNet = WTD()
    print('model initialize')

    if config.cuda: #GPU
        device = torch.device("cuda:0")
        myNet = myNet.cuda()
        myNet = torch.nn.DataParallel(myNet)

    else: #CPU
        device = torch.device("cpu")

    #myNet.apply(weight_init)
    parameters = myNet.parameters()
    optimizer =optim.Adam(parameters, lr = config.learning_rate)

    if not os.path.isdir(config.train_prediction_folder):
        os.mkdir(config.train_prediction_folder)

    iteration = 0
    train_net(myNet, device, dataloader, optimizer, iteration)