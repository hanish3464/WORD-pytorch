import torch
from collections import OrderedDict
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Hangul_Dataset
from ltr import LTR # ltr is res18 network
import time
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import opt
import file_utils
import argparse


parser = argparse.ArgumentParser(description='Text Recognition Train Process')

parser.add_argument('--save_models', default='./save/', type=str, help='saved model path')
parser.add_argument('--epoch', default=5, type=int, help='epoch')
parser.add_argument('--batch', default=64, type=int, help='batch size')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--lr_decay_step', default=5, type=int, help='decay step')
parser.add_argument('--lr_decay_gamma', default=0.1, type=float, help='decay gamma')
parser.add_argument('--train_size', default=224, type=int, help='train image size, resnet default 224')
parser.add_argument('--display_interval', default=100, type=int, help='display train log per interval')
parser.add_argument('--save_interval', default=1, type=int, help='save model interval')
parser.add_argument('--blur', action='store_true', default=False, help='data augmentation: blurring')
parser.add_argument('--distort', action='store_ture', default=False, help='data augmentation : distort')

args = parser.parse_args()

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def adjust_lr(optimizer, decay=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']


def train_net(model=None, data_loader=None, optimizer=None, epoch=50, lr=0.001, lr_decay_step=10,
              DISPLPAY_INTERVAL=None, SAVE_INTERVAL=None, lr_decay_gamma=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()

    total_step = len(data_loader)

    print('[LINE-TEXT-RECOGNITION TRAINING KICK-OFF]')
    for e in range(1, epoch + 1):
        model.train()
        start = time.time()
        if e % lr_decay_step == 0:
            adjust_lr(optimizer, lr_decay_gamma)
            lr *= lr_decay_gamma

        for k, (image, label) in enumerate(data_loader):

            image = image.to(device)
            label = label.to(device)

            y = model(image)
            loss = criterion(y, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (k + 1) % DISPLPAY_INTERVAL == 0:
                end = time.time()
                print('Epoch [{}/{}], Step [{}/{}], lr: {}, Loss: {:.4f}, TIME COST: {:.4f}'
                      .format(e, epoch, k + 1, total_step, lr, loss.item(), end - start))
                start = time.time()
        start = time.time()
        if e % SAVE_INTERVAL == 0:
            file_utils.mkdir(dir=[save_models])
            print(
                'save model ... -> {}'.format(save_models + 'res18' + '-' + str(e) + '.pth'))
            torch.save(model.state_dict(), save_models + 'res18' + '-' + repr(e) + '.pth')


def train(args):

    datasets = Hangul_Dataset(csv_path=opt.RECOGNITION_CSV_PATH, label_path='./labels-2213.txt',
                              image_size=opt.RECOG_TRAIN_SIZE, train=True, blur=False, distort=False)

    train_loader = DataLoader(dataset=datasets, batch_size=args.batch, shuffle=True, drop_last=True)

    model = LTR().cuda()
    model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_net(model=model, data_loader=train_loader, optimizer=optimizer, epoch=args.epoch,
              lr=args.lr, lr_decay_step=args.lr_decay_step, DISPLPAY_INTERVAL=args.display_interval,
              SAVE_INTERVAL=args.save_interval, lr_decay_gamma=args.lr_decay_gamma)


if __name__ == '__main__':
    train(args)
