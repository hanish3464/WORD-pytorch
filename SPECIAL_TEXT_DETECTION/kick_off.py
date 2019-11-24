"""
This Project is WTD(Webtoon Text Detection) based on NAVER CLOVA AI RESEARCH paper(CRAFT).

Future Science Technology Internship
Ajou Univ.
Writer: Han Kim

"""

import argparse
import train
import test
import evaluation

                    '''PROJECT KICK OFF'''

parser = argparse.ArgumentParser(description='Webtoon Text Localization(Detection)')

parser.add_argument('--train', default=False, type=bool, help='train flag')
parser.add_argument('--test', default=False, type=bool, help='test flag')
parser.add_argument('--evaluation', default=False, type=bool, help='evaluation flag')

args = parser.parse_args()

'''This is training part'''
if args.train: train.train()

'''This is testing part'''
if args.test: test.test()

'''This is evaluation part(under developing)'''
if args.evaluation: evaluation.evaluation()

