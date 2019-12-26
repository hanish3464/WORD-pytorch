import os, sys
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient
from model.faster_rcnn.resnet import resnet
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import argparse
import file_utils
import opt


parser = argparse.ArgumentParser(description='faster RCNN Train Process')

parser.add_argument('--save_models', default='./save/', type=str, help='saved model path')
parser.add_argument('--epoch', default=10, type=int, help='epoch')
parser.add_argument('--batch', default=4, type=int, help='batch size')
parser.add_argument('--backbone', default='res101', type=str, help='backbone : res101, vgg16')
parser.add_argument('--num_workers', default=0, type=int, help='the number of cpu core for data processing')
parser.add_argument('--lr', default=4e-3, type=float, help='learning rate')
parser.add_argument('--lr_decay_step', default=5, type=int, help='decay step')
parser.add_argument('--lr_decay_gamma', default=0.1, type=float, help='decay gamma')
parser.add_argument('--multi_gpus', action='store_true', default=False, help='whether use multi gpus')
parser.add_argument('--large_scale', action='store_true', default=False, help='whether use large imag scale')
parser.add_argument('--class_agnostic', action='store_true', default=False, help='whether to perform class_agnostic bbox regression')
parser.add_argument('--optimizer', default='sgd', type=str, help='training optimizer')
parser.add_argument('--session', default=1, type=int, help='training session')
parser.add_argument('--resume', default=False, action='store_true', help='resume checkpoint or not')
parser.add_argument('--resume_epoch', default=20, type=int, help='resume epoch point')
parser.add_argument('--resume_batch', default=8, type=int, help='resume batch size')
parser.add_argument('--use_tfboard', action='store_true', default=False, help='whether use tensorboard')
parser.add_argument('--display_interval', default=20, type=int, help='display train log per interval')

args = parser.parse_args()


class sampler(Sampler):

    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch * batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


def train(args):
    file_utils.rm_all_dir(dir='./train/cache/')  # clean cache
    dataset_name = "voc_2007_trainval"
    args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    args.cfg_file = "cfgs/{}_ls.yml".format(args.backbone) if args.large_scale else "cfgs/{}.yml".format(args.backbone)

    if args.cfg_file is not None: cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None: cfg_from_list(args.set_cfgs)

    np.random.seed(cfg.RNG_SEED)

    cfg.TRAIN.USE_FLIPPED = opt.BUBBLE_TRAIN_FLIP
    cfg.USE_GPU_NMS = opt.cuda

    _, _, _, name_lists = file_utils.get_files('./train/images/')
    file_utils.makeTrainIndex(names=name_lists, save_to='./train/trainval.txt')
    imdb, roidb, ratio_list, ratio_index = combined_roidb(dataset_name)
    train_size = len(roidb)

    print('TRAIN IMAGE NUM: {:d}'.format(len(roidb)))

    file_utils.mkdir(dir=[args.save_models])

    sampler_batch = sampler(train_size, args.batch)

    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch,\
                             imdb.num_classes, training=True)

    dataloader = DataLoader(dataset, batch_size=args.batch,
                            sampler=sampler_batch, num_workers=args.num_workers)

    im_data = Variable(torch.FloatTensor(1).cuda())
    im_info = Variable(torch.FloatTensor(1).cuda())
    num_boxes = Variable(torch.LongTensor(1).cuda())
    gt_boxes = Variable(torch.FloatTensor(1).cuda())

    fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=False)
    fasterRCNN.create_architecture()

    lr = args.lr

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1),\
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if opt.cuda:
        cfg.CUDA = True
        fasterRCNN.cuda()

    if args.resume:
        load_name = os.path.join(args.save_models,
                                 'Speech-Bubble-Detector-{}-{}-{}.pth'.format(args.backbone, args.resume_epoch, args.resume_batch))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']

    if args.multi_gpus:
        fasterRCNN = nn.DataParallel(fasterRCNN)

    iters_per_epoch = int(train_size / args.batch)

    if args.use_tfboard:
        from tensorboardX import SummaryWriter

        logger = SummaryWriter("logs")

    args.max_epochs = args.epoch
    for epoch in range(1, args.epoch + 1):

        fasterRCNN.train()
        loss_temp = 0
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter = iter(dataloader)
        for step in range(iters_per_epoch):
            data = next(data_iter)
            im_data.data.resize_(data[0].size()).copy_(data[0])
            im_info.data.resize_(data[1].size()).copy_(data[1])
            gt_boxes.data.resize_(data[2].size()).copy_(data[2])
            num_boxes.data.resize_(data[3].size()).copy_(data[3])

            fasterRCNN.zero_grad()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                   + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            loss_temp += loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            if args.backbone == "vgg16":
                clip_gradient(fasterRCNN, 10.)
            optimizer.step()

            if step % args.display_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (args.display_interval + 1)

                if args.multi_gpus:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                print("[epoch %d][iter %d/%d] loss: %.4f, lr: %.2e" \
                      % (epoch, step, iters_per_epoch, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box
                    }
                    logger.add_scalars("logs_s_{}/losses".format(args.session), info,
                                       (epoch - 1) * iters_per_epoch + step)

                loss_temp = 0
                start = time.time()

        save_name = args.save_models + args.backbone + '-' + str(epoch) + '.pth'
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.multi_gpus else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': False,
        }, save_name)
        print('save model: {}'.format(save_name))

    if args.use_tfboard:
        logger.close()


train(args)
