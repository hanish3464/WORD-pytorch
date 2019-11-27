import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Hangul_Dataset
import config
from wtr import WTR
import time

def train():
    datasets = Hangul_Dataset(csv_path=config.TRAIN_CSV_PATH, label_path=config.LABEL_PATH,
                              image_size=config.TARGET_IMAGE_SIZE, train=True)

    train_loader = DataLoader(dataset=datasets, batch_size=config.BATCH, shuffle=True, drop_last=True)

    myNet = WTR().cuda()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(myNet.parameters(), lr=config.LEARING_RATE)

    total_step = len(train_loader)

    print('[WEBTOON-TEXT-RECOGNITION TRAINING KICK-OFF]')
    for epoch in range(config.EPOCH):
        st = time.time()
        for k, (image, label) in enumerate(train_loader):

            image = image.to(device)
            label = label.to(device)

            y = myNet(image)
            loss = criterion(y, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (k + 1) % 5 == 0:
                st = time.time()
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} Time Cost: {:.4f}'
                      .format(epoch + 1, config.EPOCH, k + 1, total_step, loss.item(), ed - st))
                ed = time.time()

        print('save model ... -> {}'.format(config.SAVED_MODEL_PATH + 'wtr-' + str(epoch+1) + '.pth'))
        torch.save(myNet.state_dict(), config.SAVED_MODEL_PATH + 'wtr-' + repr(epoch+1) + '.pth')


if __name__ == '__main__':
    train()
