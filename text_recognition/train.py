import torch
from collections import OrderedDict
import torch.nn as nn
from torch.utils.data import DataLoader
from text_recognition.dataset import Hangul_Dataset
import text_recognition.config as config
from text_recognition.wtr import WTR
from text_recognition.backbone import *
import time
import argparse
import text_recognition.file_utils as file_utils

#network = {'res18': ResNet18(), 'res34': ResNet34(), 'res50': ResNet50(),
#           'res101': ResNet101(), 'res152': ResNet152(), 'dpn26': DPN26(),
#           'dpn92': DPN92(), 'vgg11': VGG('VGG11'), 'vgg13': VGG('VGG13'),
#           'vgg16': VGG('VGG16'), 'vgg19': VGG('VGG19'), 'wtr': WTR()}


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
              DISPLPAY_INTERVAL=None, SAVE_INTERVAL=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()

    total_step = len(data_loader)

    print('[WEBTOON-TEXT-RECOGNITION TRAINING KICK-OFF]')
    for e in range(1, epoch + 1):
        model.train()
        start = time.time()
        if e % lr_decay_step == 0:
            adjust_lr(optimizer, config.LR_DECAY_GAMMA)
            lr *= config.LR_DECAY_GAMMA

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
                      .format(e, epoch, k + 1, total_step, config.LEARNING_RATE, loss.item(), end - start))
                start = time.time()
        start = time.time()
        if e % SAVE_INTERVAL == 0:
            print(
                'save model ... -> {}'.format(config.SAVED_MODEL_PATH + args.net + '-' + str(e) + '.pth'))
            torch.save(model.state_dict(), config.SAVED_MODEL_PATH + args.net + '-' + repr(e) + '.pth')


def train_linear_classifier(args):
    label_mapper = file_utils.makeLabelMapper(config.LABEL_PATH)
    if args.transfer:

        img_lists, _, _, _ = file_utils.get_files(config.TRANFSER_TRAIN_IMAGE_PATH)

        test_txt = [];
        test_num = []
        for txt in config.TRANSFER_CASE:
            test_num.append(label_mapper[0].tolist().index(txt))
            test_txt.append(txt)

        file_utils.createCustomCSVFile(src=config.TRANSFER_TRAIN_CSV_PATH, files=img_lists, gt=test_txt, nums=test_num)

    datasets = Hangul_Dataset(csv_path=config.TRANSFER_TRAIN_CSV_PATH, label_path=config.LABEL_PATH,
                              image_size=config.TARGET_IMAGE_SIZE, train=True)
    transfer_loader = DataLoader(dataset=datasets, batch_size=config.TRANSFER_BATCH, shuffle=False)

    model = network[args.net]
    model.load_state_dict(copyStateDict(torch.load(config.PRETRAINED_MODEL_PATH)))
    for parameter in model.parameters():
        parameter.requires_grad = False
    n_features = 512
    model.fc = nn.Linear(n_features, config.NUM_CLASSES)

    if config.CUDA:
        model = model.cuda()
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=config.TRANSFER_LEARNING_RATE)

    train_net(model=model, data_loader=transfer_loader, optimizer=optimizer, epoch=config.TRANSFER_EPOCH,
              lr=config.TRANSFER_LEARNING_RATE, lr_decay_step=config.TRANSFER_LR_DECAY_STEP, DISPLPAY_INTERVAL=10,
              SAVE_INTERVAL=1)


def train(args):
    datasets = Hangul_Dataset(csv_path=config.TRAIN_CSV_PATH, label_path=config.LABEL_PATH,
                              image_size=config.TARGET_IMAGE_SIZE, train=True)

    train_loader = DataLoader(dataset=datasets, batch_size=config.BATCH, shuffle=True, drop_last=True)

    model = WTR().cuda()
    model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    train_net(model=model, data_loader=train_loader, optimizer=optimizer, epoch=config.EPOCH,
              lr=config.LEARNING_RATE, lr_decay_step=config.LR_DECAY_STEP, DISPLPAY_INTERVAL=config.DISPLAY_INTERVAL,
              SAVE_INTERVAL=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text Recognition Test')
    parser.add_argument('--net', default='wtr', type=str, help='select model architecture')
    parser.add_argument('--transfer', action='store_true', default=False,
                        help='transfer Learning final linear Classifier')
    args = parser.parse_args()
    if args.transfer:
        train_linear_classifier(args)
    else:
        train(args)
