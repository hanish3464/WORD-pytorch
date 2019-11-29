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

    model = WTR().cuda()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARING_RATE)

    total_step = len(train_loader)

    print('[WEBTOON-TEXT-RECOGNITION TRAINING KICK-OFF]')

    model.train()

    for epoch in range(config.EPOCH):

        start = time.time()
        for k, (image, label) in enumerate(train_loader):

            image = image.to(device)
            label = label.to(device)

            y = model(image)
            loss = criterion(y, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (k + 1) % config.DIPLAY_INTERVAL == 0:
                end = time.time()
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} TIME COST: {:.4f}'
                      .format(epoch + 1, config.EPOCH, k + 1, total_step, loss.item(), end - start))

        end = time.time()

        print('save model ... -> {}'.format(config.SAVED_MODEL_PATH + 'wtr-' + str(epoch+1) + '.pth'))
        torch.save(model.state_dict(), config.SAVED_MODEL_PATH + 'wtr-' + repr(epoch+1) + '.pth')


if __name__ == '__main__':
    train()
