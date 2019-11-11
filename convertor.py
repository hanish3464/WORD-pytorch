import debug
import json
import config
import os
import argparse
import xml.etree.ElementTree as elemTree
import shutil
import file

parser = argparse.ArgumentParser(description='gt ext setting')
parser.add_argument('--xml', default=False, type=bool, help='xml flag')
parser.add_argument('--json', default=False, type=bool, help='json flag')
parser.add_argument('--copy', default=False, type=bool, help='file copy')
args = parser.parse_args()

json_gt_folder = config.json_gt_folder
if not os.path.isdir(json_gt_folder):
    os.mkdir(json_gt_folder)

json_path = config.json_path
if not os.path.isdir(json_path):
    os.mkdir(json_path)

xml_gt_folder = './conversion/xml_to_txt_gt/'
if not os.path.isdir(xml_gt_folder):
    os.mkdir(xml_gt_folder)

xml_path = './conversion/xml_gt/'

if not os.path.isdir(xml_path):
    os.mkdir(xml_path)

elif args.json:
    json_path_list = file.get_files(json_path)
    json_path_list = json_path_list[2]
    filename, file_ext = os.path.splitext(os.path.basename(json_path_list[0]))
    print(filename, file_ext)
    for i in range(0, config.gt_json_num):
        filename, file_ext = os.path.splitext(os.path.basename(json_path_list[i]))
        with open(json_gt_folder+ filename + '.txt', 'w') as txt_file:
            with open(json_path + filename + file_ext) as json_file:
                json_tmp = json.load(json_file)
                for j in range(0, len(json_tmp['objects'])):
                    a = json_tmp['objects'][j]['points']['exterior']
                    poly = [a[0][0],a[0][1],a[1][0],a[0][1],a[1][0],a[1][1],a[0][0],a[1][1]]
                    str_result = ','.join([str(p) for p in poly]) + '\r\n'
                    txt_file.write(str_result)
elif args.copy:
     shutil.copy(config.json_gt_folder + str(1) + '.txt', config.test_ground_truth)
