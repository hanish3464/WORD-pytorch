from skimage import io
import cv2
import numpy as np
import torch

def loadImage(img_file):
    img = io.imread(img_file)
    if img.shape[0] == 2:
        img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)
    return img

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    img = in_img.copy().astype(np.float32)
    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img

def cvtColorGray(img=None):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def tranformToTensor(img=None, size=None):

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite('./vis_train.jpg', img)
    img = torch.from_numpy(img).float().unsqueeze(0)

    return img