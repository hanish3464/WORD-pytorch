"""
This Project is SBD(Speech Bubble Detection) based paper(Faster RCNN).
Future Science Technology Internship
Ajou Univ.
Writer: Han Kim
"""

import argparse
#import train
import test

'''PROJECT KICK OFF'''

parser = argparse.ArgumentParser(description='Speech Bubble Localization(Detection)')

parser.add_argument('--train', default=False, type=bool, help='train flag')
parser.add_argument('--test', default=False, type=bool, help='test flag')

args = parser.parse_args()

'''This is training part'''
#if args.train: train.train()

'''This is testing part'''
if args.test: test.test()
