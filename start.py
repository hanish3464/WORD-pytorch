"""
This Project is WTD(Webtoon Text Detection) based on NAVER CLOVA AI RESEARCH paper.

Future Science Technology Internship
Ajou Univ.
Major : Software and Computer Engineering
Writer: Han Kim

"""
import argparse

import train
import test
import evaluation
import config


#ARGUMENT PARSER START
parser = argparse.ArgumentParser(description='Webtoon Text Localization(Detection)')

parser.add_argument('--train', default=False, type=bool, help='train flag')
parser.add_argument('--test', default=False, type=bool, help='test flag')
parser.add_argument('--evaluation', default=False, type=bool, help='evaluation flag')
parser.add_argument('--orig', default=False, type=bool, help = 'original image test')


args = parser.parse_args()


if args.train:
    train.train()

if args.test:
    if args.orig:
        config.test_prediction_folder = config.orig_pdt_img
        config.test_ground_truth = config.orig_ground_truth
        config.test_images_folder_path = config.orig_img
        config.test_mask =  config.orig_mask
    test.test()

if args.evaluation:
    evaluation.evaluation()
