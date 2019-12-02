from collections import OrderedDict
import torch
import config
from dataset import Hangul_Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from wtr import WTR
import time
import file_utils
import argparse


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


def test(args):
    model = WTR()
    print('Loading model from defined path :' + config.PRETRAINED_MODEL_PATH)

    if config.CUDA:
        model.load_state_dict(copyStateDict(torch.load(config.PRETRAINED_MODEL_PATH)))
        model = model.cuda()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()
    cudnn.benchmark = False

    label_mapper = file_utils.makeLabelMapper(config.LABEL_PATH)

    if args.custom:
        img_lists, _, _, _ = file_utils.get_files(config.TEST_CUSTOM_IMAGE_PATH)
        test_txt = []; test_num = []
        for txt in config.TEST_CASE:
            test_num.append(label_mapper[0].tolist().index(txt))
            test_txt.append(txt)

        file_utils.createCustomCSVFile(src=config.TEST_CUSTOM_CSV_PATH, files=img_lists, gt=test_txt, nums=test_num)
        config.TEST_CSV_PATH = config.TEST_CUSTOM_CSV_PATH

    datasets = Hangul_Dataset(csv_path=config.TEST_CSV_PATH, label_path=config.LABEL_PATH,
                              image_size=config.TARGET_IMAGE_SIZE, train=False)
    test_loader = DataLoader(dataset=datasets, batch_size=config.TEST_BATCH, shuffle=False, drop_last=False)

    t = time.time()

    with torch.no_grad():
        total = 0
        correct = 0
        x = 0
        print('---------------------------res---------------------------')
        for k, (image, label) in enumerate(test_loader):
            image = image.to(device)
            label = label.to(device)
            y = model(image)
            _, predicted = torch.max(y.data, 1)

            prob = F.softmax(y.data, dim=1)
            probability, index = torch.max(prob, 1)
            #print(probability)
            #print(probability.shape)

            total += label.size(0)
            correct += (predicted == label).sum().item()


            for pred, lab in zip(predicted, label):
                x+=1
                if label_mapper[0][pred] != label_mapper[0][lab]:
                    print("index:{} [PRED:{}/ANS:{}], ".format(x, label_mapper[0][pred], label_mapper[0][lab]))
            x = 0
    print('')
    print('---------------------------------------------------------')
    print('TOTAL LETTERS: {}'.format(x))
    print('The Test Accuracy Of The Model: {:.4f} %'.format(100 * correct / total))
    print("TOTAL TIME : {}s".format(time.time() - t))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text Recognition Test')
    parser.add_argument('--custom', default=False, action='store_true', help='select custom test dataset')
    args = parser.parse_args()
    test(args)
