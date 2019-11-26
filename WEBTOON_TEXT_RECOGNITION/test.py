from collections import OrderedDict
import torch
import config
import imgproc
import pandas as pd
import numpy as np
import cv2
from wtr import WTR
import time

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

def test():

    myNet = WTR()
    print('Loading model from defined path :' + config.PRETRAINED_MODEL_PATH)
    if config.CUDA: pass
        myNet.load_state_dict(copyStateDict(torch.load(config.PRETRAINED_MODEL_PATH)))
        myNet = myNet.cuda()

    myNet.eval()
    t = time.time()

    data = pd.read_csv(config.TEST_CSV_PATH)
    labels = np.asarray(data.iloc[:, 2])
    images = np.asarray(data.iloc[:, 0])
    print(labels, len(labels))
    print(images, len(images))


    total = 0
    correct = 0
    for k, in_path in enumerate(images):

        image = imgproc.loadImage(in_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (config.TRAIN_IMAGE_SIZE, config.TRAIN_IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

        y = myNet(image)
        _, predicted = torch.max(y.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))


    print("TOTAL TIME : {}s".format(time.time() - t))


if __name__ =='__main__':
    test()