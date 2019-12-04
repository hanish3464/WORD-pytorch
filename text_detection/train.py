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
import time

def adjust_learning_rate(optimizer, step):
    lr = config.lr * (0.8 ** step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train():
    ''''''

    ''' MAKE DATASET '''
    datasets = webtoon_dataset(config.TRAIN_IMAGE_PATH, config.TRAIN_LABEL_PATH, config.TRAIN_IMAGE_SIZE)
    train_data_loader = DataLoader(datasets, batch_size=config.BATCH, shuffle=True)

    ''' INITIALIZE MODEL, GPU, OPTIMIZER, and, LOSS '''

    model = WTD()
    if config.cuda:
        model = model.cuda()
        model = torch.nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = WTD_LOSS()

    ''' SET PATH '''
    if not os.path.isdir(config.TRAIN_PREDICTION_PATH):  os.mkdir(config.TRAIN_PREDICTION_PATH)

    step_idx = 0

    ''' KICK OFF TRAINING PROCESS '''
    for e in range(config.EPOCH):

        start = time.time()
        ''' LOAD MATERIAL FOR TRAINING FROM DATALOADER '''
        for idx, (image, region_score_GT, affinity_score_GT, confidence) in enumerate(train_data_loader):

            ''' ADJUST LEARNING RATE PER 20000 ITERATIONS '''
            if idx % config.LR_DECAY_STEP == 0 and idx != 0:
                step_idx += 1
                adjust_learning_rate(optimizer, step_idx)

            ''' CONVERT NUMPY => TORCH '''
            images = Variable(image.type(torch.FloatTensor)).cuda()
            region_score_GT = Variable(region_score_GT.type(torch.FloatTensor)).cuda()
            affinity_score_GT = Variable(affinity_score_GT.type(torch.FloatTensor)).cuda()
            confidence = Variable(confidence.type(torch.FloatTensor)).cuda()

            ''' PASS THE MODEL AND PREDICT SCORES '''
            y, _ = model(images)
            score_region = y[:, :, :, 0].cuda()
            score_affinity = y[:, :, :, 1].cuda()

            if config.VIS:
                if idx % 20 == 0 and idx != 0:
                    for idx2 in range(config.BATCH):
                        render_img1 = score_region[idx2].cpu().detach().numpy().copy() * 255.0
                        render_img2 = score_affinity[idx2].cpu().detach().numpy().copy() * 255.0
                        render_img = np.hstack((render_img1, render_img2))
                        if not os.path.isdir('./vis/mask/epoch' + str(e) + '/'):
                            os.mkdir('./vis/mask/epoch' + str(e) + '/')
                        cv2.imwrite('./vis/mask/epoch' + str(e) + '/mask_' + str(idx) + '_' + str(idx2) + '.jpg', render_img)


            ''' CALCULATE LOSS VALUE AND UPDATE WEIGHTS '''
            optimizer.zero_grad()
            loss = criterion(region_score_GT, affinity_score_GT, score_region, score_affinity, confidence)
            loss.backward()
            optimizer.step()

            if idx % config.DISPLAY_INTERVAL == 0:
                end = time.time()
                print('epoch: {}, iter:[{}/{}], loss: {}, Time Cost: {}s'.format(i, idx, len(train_data_loader), loss.item(), end-start))
                start = time.time()

        ''' SAVE MODEL PER 2 EPOCH '''
        start = time.time()
        if e % config.SAVE_INTERVAL == 0:
            print('save model ... :' + config.SAVED_MODEL_PATH)
            torch.save(myNet.module.state_dict(), config.SAVED_MODEL_PATH + 'wtd' + repr(e) + '.pth')
