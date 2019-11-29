import codecs
import os
import numpy as np

def saveCSV(dir=None, dst=None, index=None, label=None, num=None):
    distorted_image_file = dir + str(index) + '.jpeg'
    dst.write(u'{},{},{}\n'.format(distorted_image_file, label, num))

def createCustomCSVFile(src=None, files=None, gt=None, nums=None):
    labels_csv = codecs.open(os.path.join(src), 'w', encoding='utf-8')
    for k, file in enumerate(files):
        labels_csv.write(u'{},{},{}\n'.format(file, gt[k-1], nums[k-1]))

def loadText(txt_file):
    arr = []
    with codecs.open(txt_file, encoding='utf-8_sig') as file:
        lines= file.readlines()
        for line in lines: arr.append(line.strip('\r\n'))
    return arr

def loadSpacingWordInfo(txt_file):
    arr_list = list()
    length = 0
    with codecs.open(txt_file, encoding='utf-8_sig') as file:
        coordinate = file.readlines()
        for line in coordinate:
            tmp = line.split(',')
            tmp[-1] = tmp[-1].strip('\r\n')
            if tmp[0] == '': continue
            arr_coordinate = [int(n) for n in tmp]
            coordinate = np.array(arr_coordinate).tolist()
            arr_list.append(coordinate)
            length += 1
    return arr_list, length

def makeLabelMapper(in_path):
    label_map = loadText(in_path)
    label_num = np.arange(len(label_map))
    label_mapper = np.vstack((np.array(label_map), label_num))
    return label_mapper

def saveImage(dir=None, img=None, index=None):
    distorted_image_file = dir + str(index) + '.jpeg'
    img.save(distorted_image_file, 'JPEG')

def get_files(img_dir):
    imgs, masks, xmls, names = list_files(img_dir)
    return imgs, masks, xmls, names

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    names = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            names.append(filename)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt' or ext =='.json':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    img_files.sort()
    mask_files.sort()
    gt_files.sort()
    names.sort()
    return img_files, mask_files, gt_files, names