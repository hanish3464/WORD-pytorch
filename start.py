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
import char_generator
import word_generator
import config


#ARGUMENT PARSER START
parser = argparse.ArgumentParser(description='Webtoon Text Localization(Detection)')

parser.add_argument('--train', default=False, type=bool, help='train flag')
parser.add_argument('--test', default=False, type=bool, help='test flag')
parser.add_argument('--evaluation', default=False, type=bool, help='evaluation flag')
parser.add_argument('--orig', default=False, type=bool, help = 'original image test')
parser.add_argument('--char_gen', default= False, type = bool, help = 'both char and word text annotation is divided from psd file')
parser.add_argument('--word_gen', default= False, type = bool, help = 'both word text annotation is divided from psd file')

parser.add_argument('--webtoon', default=False, type= bool, help ='webtoon Dataset')
parser.add_argument('--synth', default=False, type= bool, help ='synthText Dataset')
args = parser.parse_args()


if args.train:
    '''This is training part'''
    if args.synth: config.data_options = 'synthText'
    train.train()

if args.test:
    '''This is test part'''
    if args.orig:
        '''This configuration selects origin folder path'''
        config.test_prediction_folder = config.orig_pdt_img
        config.test_ground_truth = config.orig_ground_truth
        config.test_images_folder_path = config.orig_img
        config.test_mask =  config.orig_mask
    test.test()

if args.evaluation:
    '''This is evaluation part'''
    evaluation.evaluation()

if args.char_gen:
    '''This is divided psd part '''
    char_generator.pseudo_gt_generator()
if args.word_gen:
    word_generator.word_gt_generator()
