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



#ARGUMENT PARSER START
parser = argparse.ArgumentParser(description='Webtoon Text Localization(Detection)')

parser.add_argument('--train', default=False, type=bool, help='train flag')
parser.add_argument('--test', default=False, type=bool, help='test flag')
parser.add_argument('--evaluation', default=False, type=bool, help='evaluation flag')


args = parser.parse_args()


if args.train:
    train.train()

if args.test:
    test.test()

if args.evaluation:
    evaluation.evaluation()
