"""train.py"""

import cv2
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import  RandomSampler, WeightedRandomSampler
from torch.autograd import Variable
import os
import torch.backends.cudnn as cudnn
from wtd import WTD
import config
from loss import WTD_LOSS
from dataset import webtoon_dataset
import time
import numpy as np
import sys

def adjust_learning_rate(optimizer, step):
    lr = config.lr * (0.8 ** step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def select_dataset(opt = None):
    if opt == 'synthText': pass
    if opt == 'webtoon':
        datasets = webtoon_dataset(config.train_images_path, config.train_char_gt_path, config.train_word_gt_path,
                                   config.train_image_size)
        train_data_loader = DataLoader(datasets, batch_size=config.batch, shuffle=False, num_workers=0, drop_last=False,
                                       pin_memory=False)
    return train_data_loader

def train():
    """there is under developing"""

    train_data_loader = select_dataset(opt = config.data_options)

    myNet = WTD()

    if config.cuda:  # GPU
        myNet = myNet.cuda()
        #myNet = torch.nn.DataParallel(myNet).cuda()

    optimizer = optim.Adam(myNet.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = WTD_LOSS()

    if not os.path.isdir(config.train_prediction_path):  os.mkdir(config.train_prediction_path)

    # myNet.train()
    loss_time = 0
    step_idx = 0
    loss_value = 0
    for i in range(config.epoch):
        #st = time.time()
        loss_value = 0
        for idx, (image, region_score_GT, affinity_score_GT, confidence) in enumerate(train_data_loader):

            if idx % 20000 == 0 and idx != 0:
                step_idx += 1
                adjust_learning_rate(optimizer, step_idx)

            images = Variable(image.type(torch.FloatTensor)).cuda()
            region_score_GT = Variable(region_score_GT.type(torch.FloatTensor)).cuda()
            affinity_score_GT = Variable(affinity_score_GT.type(torch.FloatTensor)).cuda()
            confidence = Variable(confidence.type(torch.FloatTensor)).cuda()

            y, _ = myNet(images)

            score_region = y[:, :, :, 0].cuda()
            score_affinity = y[:, :, :, 1].cuda()

            if i % 200 == 0 and i != 0:
                for idx2 in range(config.batch):
                    render_img1 = score_region[idx2].cpu().detach().numpy().copy() * 255.0
                    render_img2 = score_affinity[idx2].cpu().detach().numpy().copy() * 255.0
                    render_img = np.hstack((render_img1, render_img2))
                    if not os.path.isdir('./train/mask/epoch' + str(i) + '/'):
                        os.mkdir('./train/mask/epoch' + str(i) + '/')
                    cv2.imwrite('./train/mask/epoch' +str(i)+'/mask_' + str(i) + '_' + str(idx) + '.jpg', render_img)


            '''loss function'''

            optimizer.zero_grad()

            loss = criterion(region_score_GT, affinity_score_GT, score_region, score_affinity, confidence)
            loss.backward()
            optimizer.step()
            loss_value += loss.item()
            # if idx % 2 == 0 and idx > 0:
            #     et = time.time()
            #     #print('epoch {}:({}/{}) batch || training time for 2 batch {} || training loss {} ||'.format(i, idx,len(train_data_loader),et - st,loss_value / 2))
            #     loss_time = 0
            #     loss_value = 0
            #     st = time.time()

            if idx % 500 == 0 and idx != 0:
                torch.save(myNet.module.state_dict(), config.saving_model + 'wtd' + repr(idx) + '.pth')

        print('epoch: {} || training loss {} ||'.format(i, loss_value / 2))