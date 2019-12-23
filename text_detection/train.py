"""train.py"""

import cv2
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from ltd import LTD
from loss import LTD_LOSS
from dataset import webtoon_dataset
import numpy as np
import time
import imgproc
import file_utils
import argparse
import opt


parser = argparse.ArgumentParser(description='Text Detection Train Process')

parser.add_argument('--save_models', default='./save/', type=str, help='saved model path')
parser.add_argument('--epoch', default=10, type=int, help='epoch')
parser.add_argument('--batch', default=1, type=int, help='batch size')
parser.add_argument('--lr', default=4e-3, type=float, help='learning rate')
parser.add_argument('--lr_decay_step', default=5, type=int, help='decay step')
parser.add_argument('--lr_decay_gamma', default=0.8, type=float, help='decay gamma')
parser.add_argument('--train_size', default=512, type=int, help='train image size, resnet default 224')
parser.add_argument('--display_interval', default=100, type=int, help='display train log per interval')
parser.add_argument('--save_interval', default=1, type=int, help='save model interval')
parser.add_argument('--rotate', action='store_true', default=False, help='data augmentation : rotate')
parser.add_argument('--flip', action='store_true', default=False, help='data augmentation : flip')
parser.add_argument('--crop', action='store_true', default=False, help='data augmentation : crop')
parser.add_argument('--vis_train', action='store_true',default=False, help='model prediction visualization')
parser.add_argument('--region', default=0.3, type=float, help='gaussian heatmap labeling region scope')
parser.add_argument('--affinity', default=0.25, type=float, help='gaussian heatmap labeling affinity scope')


args = parser.parse_args()
opt.flip = args.flip
opt.rotate = args.rotate
opt.crop = args.crop
opt.gaussian_region = args.region
opt.gaussian_affinity = args.affinity


def adjust_learning_rate(optimizer, lr, step):
    lr = lr * (0.8 ** step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(args):

    file_utils.mkdir(dir=[args.save_models])
    if args.vis_train: file_utils.mkdir(dir=['./vis/'])

    ''' MAKE DATASET '''
    datasets = webtoon_dataset(opt.DETECTION_TRAIN_IMAGE_PATH, opt.DETECTION_TRAIN_LABEL_PATH, args.train_size)
    train_data_loader = DataLoader(datasets, batch_size=args.batch, shuffle=True)

    ''' INITIALIZE MODEL, GPU, OPTIMIZER, and, LOSS '''

    model = LTD()
    model = torch.nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lr_decay_gamma)
    criterion = LTD_LOSS()

    step_idx = 0

    model.train()
    print('TEXT DETECTION TRAINING KICK-OFF]')

    ''' KICK OFF TRAINING PROCESS '''
    for e in range(args.epoch):

        start = time.time()

        ''' LOAD MATERIAL FOR TRAINING FROM DATALOADER '''
        for idx, (image, region_score_GT, affinity_score_GT, confidence) in enumerate(train_data_loader):

            ''' ADJUST LEARNING RATE PER 20000 ITERATIONS '''
            if idx % args.lr_decay_step == 0 and idx != 0:
                step_idx += 1
                #adjust_learning_rate(optimizer, args.lr, step_idx)

            ''' CONVERT NUMPY => TORCH '''
            images = Variable(image.type(torch.FloatTensor)).cuda()
            region_score_GT = Variable(region_score_GT.type(torch.FloatTensor)).cuda()
            affinity_score_GT = Variable(affinity_score_GT.type(torch.FloatTensor)).cuda()
            confidence = Variable(confidence.type(torch.FloatTensor)).cuda()

            ''' PASS THE MODEL AND PREDICT SCORES '''
            y, _ = model(images)
            score_region = y[:, :, :, 0].cuda()
            score_affinity = y[:, :, :, 1].cuda()

            if args.vis_train:
                if idx % 20 == 0 and idx != 0 and e % 2 == 0:
                    for idx2 in range(args.batch):
                        render_img1 = score_region[idx2].cpu().detach().numpy().copy()
                        render_img2 = score_affinity[idx2].cpu().detach().numpy().copy()
                        render_img = np.hstack((render_img1, render_img2))
                        render_img = imgproc.cvt2HeatmapImg(render_img)
                        cv2.imwrite('./vis/e' + str(e) + '-s' + str(idx) + '-' + str(idx2) + '.jpg',
                                    render_img)

            ''' CALCULATE LOSS VALUE AND UPDATE WEIGHTS '''
            optimizer.zero_grad()
            loss = criterion(region_score_GT, affinity_score_GT, score_region, score_affinity, confidence)
            loss.backward()
            optimizer.step()

            if idx % args.display_interval == 0:
                end = time.time()
                print('epoch: {}, iter:[{}/{}], lr:{}, loss: {:.8f}, Time Cost: {:.4f}s'.format(e, idx,
                                                                                                len(train_data_loader),
                                                                                                args.lr,
                                                                                                loss.item(),
                                                                                                end - start))
                start = time.time()

        ''' SAVE MODEL PER 2 EPOCH '''
        start = time.time()
        if e % args.save_interval == 0:
            print('save model ... :' + args.save_models)
            torch.save(model.module.state_dict(), args.save_models + 'ltd' + repr(e) + '.pth')


train(args)