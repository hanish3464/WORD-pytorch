"""train.py"""

import cv2
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import torch.backends.cudnn as cudnn
from wtd import WTD
import config
from loss import WTD_LOSS
from dataset import webtoon_dataset
import time
import numpy as np

def adjust_learning_rate(optimizer, step):
    lr = config.lr * (0.8 ** step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train():
    """there is under developing"""

    datasets = webtoon_dataset(config.train_images_path, config.train_char_gt_path, config.train_word_gt_path,
                               config.train_image_size)
    train_data_loader = DataLoader(datasets, batch_size=config.batch, shuffle=True, num_workers=0,
                                   drop_last=True, pin_memory=True)
    myNet = WTD()

    if config.cuda:  # GPU
        myNet = myNet.cuda()
    else:  # CPU
        device = torch.device("cpu")

    optimizer = optim.Adam(myNet.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    cudnn.benchmark = True
    criterion = WTD_LOSS()

    if not os.path.isdir(config.train_prediction_path):  os.mkdir(config.train_prediction_path)

    # myNet.train()
    loss_time = 0
    step_idx = 0
    loss_value = 0
    for i in range(config.epoch):

        print('epoch :{} entered'.format(i))

        for idx, (image, region_score_GT, affinity_score_GT) in enumerate(train_data_loader):
            print(len(train_data_loader))

            if idx % 20000 == 0 and idx != 0:
                step_idx += 1
                adjust_learning_rate(optimizer, step_idx)

            images = image.type(torch.FloatTensor)
            print(images.shape)
            region_score_GT = region_score_GT.type(torch.FloatTensor)
            affinity_score_GT = affinity_score_GT.type(torch.FloatTensor)
            images = Variable(images).cuda()
            region_score_GT = Variable(region_score_GT).cuda()
            affinity_score_GT = Variable(affinity_score_GT).cuda()
            confidence = np.ones((region_score_GT.shape[0], region_score_GT.shape[1],region_score_GT.shape[2]), np.float32)
            confidence = torch.from_numpy(confidence).float()
            confidence = Variable(confidence.type(torch.FloatTensor)).cuda()
            y, _ = myNet(images)

            score_region = y[:, :, :, 0].cuda() # what is y dimension
            score_affinity = y[:, :, :, 1].cuda()
            for idx2 in range(config.batch):
                render_img1 = score_region[idx2].cpu().detach().numpy().copy() * 255.0
                render_img2 = score_affinity[idx2].cpu().detach().numpy().copy() * 255.0
                render_img = np.hstack((render_img1, render_img2))
                cv2.imwrite('./train/mask/mask_' + str(idx) + '_' + str(idx2) + '.jpg', render_img)


            '''loss function'''

            optimizer.zero_grad()

            loss = criterion(region_score_GT, affinity_score_GT, score_region, score_affinity, confidence)
            loss.backward()
            optimizer.step()
            loss_value += loss.item()
            print(idx)
            if idx % 2 == 0 and idx > 0:
                et = time.time()
                print('epoch {}:({}/{}) batch || training time for 2 batch {} || training loss {} ||'.format(i, idx,len(train_data_loader),et - st,loss_value / 2))
                loss_time = 0
                loss_value = 0
                st = time.time()

            if idx % 5000 == 0 and idx != 0:
                torch.save(myNet.module.state_dict(), config.saving_model + 'wtd' + repr(idx) + '.pth')
