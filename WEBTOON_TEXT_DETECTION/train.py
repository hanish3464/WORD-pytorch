"""train.py"""

import cv2
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os

from wtd import WTD
import config
from loss import WTD_LOSS
from dataset import webtoon_dataset
import numpy as np

def adjust_learning_rate(optimizer, step):
    lr = config.lr * (0.8 ** step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train():
    ''''''

    ''' MAKE DATASET '''
    datasets = webtoon_dataset(config.TRAIN_IMAGE_PATH, config.TRAIN_CHAR_GT_PATH, config.TRAIN_WORD_GT_PATH, config.TRAIN_IMAGE_SIZE)
    train_data_loader = DataLoader(datasets, batch_size=config.BATCH, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)


    ''' INITIALIZE MODEL, GPU, OPTIMIZER, and, LOSS '''
    myNet = WTD()
    if config.cuda:
        myNet = myNet.cuda()
        myNet = torch.nn.DataParallel(myNet).cuda()
    optimizer = optim.Adam(myNet.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = WTD_LOSS()

    ''' SET PATH '''
    if not os.path.isdir(config.TRAIN_PREDICTION_PATH):  os.mkdir(config.TRAIN_PREDICTION_PATH)

    step_idx = 0

    ''' KICK OFF TRAINING PROCESS '''
    for i in range(config.EPOCH):

        loss_value = 0

        ''' LOAD MATERIAL FOR TRAINING FROM DATALOADER '''
        for idx, (image, region_score_GT, affinity_score_GT, confidence) in enumerate(train_data_loader):

            ''' ADJUST LEARNING RATE PER 20000 ITERATIONS '''
            if idx % 20000 == 0 and idx != 0:
                step_idx += 1
                adjust_learning_rate(optimizer, step_idx)

            ''' CONVERT NUMPY => TORCH '''
            images = Variable(image.type(torch.FloatTensor)).cuda()
            region_score_GT = Variable(region_score_GT.type(torch.FloatTensor)).cuda()
            affinity_score_GT = Variable(affinity_score_GT.type(torch.FloatTensor)).cuda()
            confidence = Variable(confidence.type(torch.FloatTensor)).cuda()

            ''' PASS THE MODEL AND PREDICT SCORES '''
            y, _ = myNet(images)
            score_region = y[:, :, :, 0].cuda()
            score_affinity = y[:, :, :, 1].cuda()


            if config.vis:
                if i % 200 == 0 and i != 0:
                    for idx2 in range(config.BATCH):
                        render_img1 = score_region[idx2].cpu().detach().numpy().copy() * 255.0
                        render_img2 = score_affinity[idx2].cpu().detach().numpy().copy() * 255.0
                        render_img = np.hstack((render_img1, render_img2))
                        if not os.path.isdir('./vis/mask/epoch' + str(i) + '/'):
                            os.mkdir('./vis/mask/epoch' + str(i) + '/')
                        cv2.imwrite('./vis/mask/epoch' +str(i)+'/mask_' + str(i) + '_' + str(idx) + '.jpg', render_img)


            ''' CALCULATE LOSS VALUE AND UPDATE WEIGHTS '''
            optimizer.zero_grad()
            loss = criterion(region_score_GT, affinity_score_GT, score_region, score_affinity, confidence)
            loss.backward()
            optimizer.step()
            loss_value += loss.item()

            if idx % 2 == 0 and idx > 0:
                 print('epoch {}:({}/{}) batch || training loss {} ||'.format(i, idx, len(train_data_loader),loss_value / 2))
                 loss_value = 0

            ''' SAVE MODEL PER 5000 ITERATIONS '''
            if idx % 5000 == 0 and idx != 0:
                torch.save(myNet.module.state_dict(), config.SAVED_MODEL_PATH + 'wtd' + repr(idx) + '.pth')
