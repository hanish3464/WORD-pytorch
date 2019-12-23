'''net_utils.py'''

import torch
import numpy as np
from object_detection.lib.model.faster_rcnn.resnet import resnet
from object_detection.lib.model.utils.config import cfg
from text_detection.ltd import LTD
from text_recognition.ltr import LTR
from collections import OrderedDict
import opt


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


def load_speech_bubble_detector(args):
    np.random.seed(opt.RNG_SEED)
    labels = opt.LABEL

    layerNum = 101
    if opt.BACKBONE == 'res152': layerNum = 152

    fasterRCNN = resnet(labels, layerNum, pretrained=False, class_agnostic=False)
    fasterRCNN.create_architecture()

    print('Loading weights from checkpoint : ({})'.format(args.object_detector))
    if opt.cuda:
        model = torch.load(args.object_detector)
    else:
        model = torch.load(args.object_detector, map_location=(lambda storage, loc: storage))

    fasterRCNN.load_state_dict(model['model'])
    if 'pooling_mode' in model.keys(): cfg.POOLING_MODE = model['pooling_mode']

    if opt.cuda:
        fasterRCNN.cuda()
    fasterRCNN.eval()
    return fasterRCNN


def load_text_detector(args):

    text_detector = LTD()
    print('Loading weights from checkpoint : ({})'.format(args.text_detector))
    text_detector.load_state_dict(copyStateDict(torch.load(args.text_detector)))
    text_detector = text_detector.cuda()
    text_detector.eval()
    return text_detector


def load_text_recognizer(args):
    text_recognizer = LTR()
    print('Loading weights from checkpoint : ({})'.format(args.text_recognizer))
    text_recognizer.load_state_dict(copyStateDict(torch.load(args.text_recognizer)))
    text_recognizer = text_recognizer.cuda()
    text_recognizer.eval()
    return text_recognizer


def load_net(args):
    models = {}
    if args.object:
        speech_bubble_detector = load_speech_bubble_detector(args)
        text_detector = load_text_detector(args)
        models.update({'bubble_detector': speech_bubble_detector, 'text_detector': text_detector})

    if args.ocr:
        text_recognizer = load_text_recognizer(args)
        models.update({'text_recognizer': text_recognizer})

    return models
