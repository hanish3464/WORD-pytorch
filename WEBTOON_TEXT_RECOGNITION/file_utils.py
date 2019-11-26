import codecs
import os

def saveCSV(dir=None, dst=None, index=None, label=None, num=None):
    distorted_image_file = dir + str(index) + '.jpeg'
    dst.write(u'{},{},{}\n'.format(distorted_image_file, label, num))

def saveImage(dir=None, img=None, index=None):
    distorted_image_file = dir + str(index) + '.jpeg'
    img.save(distorted_image_file, 'JPEG')

def loadText(txt_file):
    arr = []
    with codecs.open(txt_file, encoding='utf-8_sig') as file:
        lines= file.readlines()
        for line in lines: arr.append(line.strip('\r\n'))
    return arr

def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
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
    return img_files, mask_files, gt_files