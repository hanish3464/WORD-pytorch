import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Hangul_Dataset
import config
from wtr import WTR
from backbone import *
import time
import argparse


def adjust_lr(optimizer, decay=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']


def train(args):
    datasets = Hangul_Dataset(csv_path=config.TRAIN_CSV_PATH, label_path=config.LABEL_PATH,
                              image_size=config.TARGET_IMAGE_SIZE, train=True)

    train_loader = DataLoader(dataset=datasets, batch_size=config.BATCH, shuffle=True, drop_last=True)

    network = {'res18': ResNet18(), 'res34': ResNet34(), 'res50': ResNet50(),
               'res101': ResNet101(), 'res152': ResNet152(), 'dpn26': DPN26(),
               'dpn92': DPN92(), 'vgg11': VGG('VGG11'), 'vgg13': VGG('VGG13'),
               'vgg16': VGG('VGG16'), 'vgg19': VGG('VGG19'), 'wtr': WTR()}

    model = network[args.net].cuda()
    model = nn.DataParallel(model)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    total_step = len(train_loader)

    print('[WEBTOON-TEXT-RECOGNITION TRAINING KICK-OFF]')

    model.train()

    for epoch in range(1, config.EPOCH + 1):

        start = time.time()
        if epoch % config.LR_DECAY_STEP == 0:
            adjust_lr(optimizer, config.LR_DECAY_GAMMA)
            config.LEARNING_RATE *= config.LR_DECAY_GAMMA

        for k, (image, label) in enumerate(train_loader):

            image = image.to(device)
            label = label.to(device)

            y = model(image)
            loss = criterion(y, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (k + 1) % config.DISPLAY_INTERVAL == 0:
                end = time.time()
                print('Epoch [{}/{}], Step [{}/{}], lr: {}, Loss: {:.4f}, TIME COST: {:.4f}'
                      .format(epoch, config.EPOCH, k + 1, total_step, config.LEARNING_RATE, loss.item(), end - start))
                start = time.time()
        start = time.time()

        print('save model ... -> {}'.format(config.SAVED_MODEL_PATH + 'wtr-' + args.net + '-' + str(epoch) + '.pth'))
        torch.save(model.state_dict(), config.SAVED_MODEL_PATH + 'wtr-' + args.net + '-' + repr(epoch) + '.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text Recognition Test')
    parser.add_argument('--net', default='wtr', type=str, help='select model architecture')
    args = parser.parse_args()
    train(args)
