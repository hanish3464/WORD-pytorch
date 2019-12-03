import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Hangul_Dataset
import config
from wtr import WTR
from backbone import dpn, PreActResNet, VGG
import time

def adjust_lr(optimizer, decay=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']

def train():
    datasets = Hangul_Dataset(csv_path=config.TRAIN_CSV_PATH, label_path=config.LABEL_PATH,
                              image_size=config.TARGET_IMAGE_SIZE, train=True)

    train_loader = DataLoader(dataset=datasets, batch_size=config.BATCH, shuffle=True, drop_last=True)

    #model = WTR().cuda()
    model = dpn.DPN26().cuda()
    #model = PreActResNet.PreActResNet18().cuda()
    #model = VGG.VGG('VGG16').cuda()
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

        print('save model ... -> {}'.format(config.SAVED_MODEL_PATH + 'wtr-DPN26-' + str(epoch) + '.pth'))
        torch.save(model.state_dict(), config.SAVED_MODEL_PATH + 'wtr-DPN26-' + repr(epoch) + '.pth')


if __name__ == '__main__':
    train()
