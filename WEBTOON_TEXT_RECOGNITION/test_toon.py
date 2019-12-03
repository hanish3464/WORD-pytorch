from collections import OrderedDict
import torch
import config
import torch.backends.cudnn as cudnn
import time
import file_utils
import argparse
import imgproc
import wtr_utils
from backbone import *
from wtr import WTR


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


def pipeline_test(args):
    network = {'res18': ResNet18(), 'res34': ResNet34(), 'res50': ResNet50(), 'res101': ResNet101(),
               'res152': ResNet152(), 'dpn26': DPN26(), 'dpn92': DPN92(), 'vgg11': VGG('VGG11'), 'vgg13': VGG('VGG13'),
               'vgg16': VGG('VGG16'), 'vgg19': VGG('VGG19'), 'wtr': WTR()}

    model = network[args.net]
    if args.net == 'wtr': config.TARGET_IMAGE_SIZE = 224
    print('Loading model from defined path :' + config.PRETRAINED_MODEL_PATH)

    if config.CUDA:
        model.load_state_dict(copyStateDict(torch.load(config.PRETRAINED_MODEL_PATH)))
        model = model.cuda()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()
    cudnn.benchmark = False

    label_mapper = file_utils.makeLabelMapper(config.LABEL_PATH)
    spacing_words, _ = file_utils.loadSpacingWordInfo(config.SPACING_WORD_PATH)

    t = time.time()

    with torch.no_grad():
        image_name_nums = []
        res = []
        img_lists, _, _, name_list = file_utils.get_files(config.TEST_RECOG_PATH)
        for name in name_list: image_name_nums.append(name.split('_')[0])

        print('------------------------------OCR RESULT----------------------------------')
        for k, in_path in enumerate(img_lists):
            image = imgproc.loadImage(in_path)
            image = imgproc.cvtColorGray(image)
            image = imgproc.tranformToTensor(image, config.TARGET_IMAGE_SIZE).unsqueeze(0)
            image = image.to(device)
            y = model(image)
            _, pred = torch.max(y.data, 1)
            res.append(label_mapper[0][pred])

        wtr_utils.DISLPLAY_STDOUT(chars=res, space=spacing_words, img_name=image_name_nums, MODE=args.mode,
                                  Net=args.net)

    print("TOTAL TIME : {}s".format(time.time() - t))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text Recognition Test')
    parser.add_argument('--mode', default='all', type=str, help='opt: stdout, file, all')
    parser.add_argument('--net', default='wtr', type=str, help='select model architecture')
    args = parser.parse_args()
    pipeline_test(args)
