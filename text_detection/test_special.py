import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import torch.backends.cudnn as cudnn
import cv2
import config
from wtd import WTD
import torch.nn as nn
import file_utils
import imgproc
import sys
from collections import OrderedDict
import wtd_utils

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


def test_net(model, image, text_threshold, link_threshold, low_text, cuda, k):
    #img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, config.MAXIMUM_IMAGE_SIZE,
    #                                                                     interpolation=cv2.INTER_LINEAR,
    #                                                                      mag_ratio=config.MAG_RATIO)
    #ratio_h = ratio_w = 1 / target_ratio
    copy = image.copy()
    image = imgproc.normalizeMeanVariance(image)
    x = torch.tensor(image).float().permute(2, 0, 1)
    x = Variable(x.type(torch.FloatTensor))
    x = Variable(x.unsqueeze(0))

    if cuda: x = x.cuda()

    y, _ = model(x)

    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    wtd_utils.getdetReigion_core(copy, score_text, score_link, text_threshold, link_threshold, low_text, k)

    cv2.imwrite('./res/text_' + str(k) + '.jpg', imgproc.cvt2HeatmapImg(score_text))
    cv2.imwrite('./res/link_' + str(k) + '.jpg', imgproc.cvt2HeatmapImg(score_link))


def test_sounds_effect():
    sound_effect_detector = WTD()
    print('Loading model from defined path :' + config.PRETRAINED_MODEL_PATH)
    if config.cuda:
        sound_effect_detector.load_state_dict(copyStateDict(torch.load(config.PRETRAINED_MODEL_PATH)))
    else:
        sound_effect_detector.load_state_dict(
            copyStateDict(torch.load(config.PRETRAINED_MODEL_PATH, map_location='cpu')))

    if config.cuda:
        sound_effect_detector.cuda()
        sound_effect_detector = nn.DataParallel(sound_effect_detector)

    #sound_effect_detector.eval()
    print('[SOUND EFFECT DETECTOR TEST KICK-OFF]')

    img_list, _, _, name_list = file_utils.get_files(config.TRAIN_IMAGE_PATH)

    for k, in_path in enumerate(img_list):
        sys.stdout.write('TEST IMAGES: {:d}/{:d}: {:s} \r'.format(k + 1, len(img_list), in_path))
        sys.stdout.flush()
        img = imgproc.loadImage(in_path)
        cv2.imwrite('./test.jpg', img)
        test_net(sound_effect_detector, img, config.text_threshold,
                 config.link_threshold, config.low_text, config.cuda, k)


if __name__ == '__main__':
    test_sounds_effect()
