import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import hangul_dataset
import config
from wtr import WTR

def train():

    datasets = hangul_dataset(config.TRAIN_CSV_PATH, config.LABEL_PATH, config.TRAIN_IMAGE_SIZE)
    train_loader = DataLoader(dataset=datasets, batch_size=config.BATCH, shuffle=True, drop_last=True)

    myNet = WTR().cuda()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(myNet.parameters(), lr=config.LEARING_RATE)

    total_step = len(train_loader)
    print('[WEBTOON-TEXT-RECOGNITION TRAINING KICK-OFF]')
    for epoch in range(config.EPOCH):

        for k, (image, label) in enumerate(train_loader):

            image = image.to(device)
            label = label.to(device)
            y = myNet(image)
            loss = criterion(y, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (k+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, config.EPOCH, k + 1, total_step, loss.item()))
        print('save model ... -> {}'.format(config.SAVED_MODEL_PATH + 'wtr-' + str(epoch) + '.pth'))
        torch.save(myNet.module.state_dict(), config.SAVED_MODEL_PATH + 'wtr-' + repr(epoch) + '.pth')


if __name__ == '__main__':
    train()