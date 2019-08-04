import sys
import matplotlib.pyplot as plt
from matplotlib.image import imread
import skimage as io
import numpy as np
def printing(img):
    plt.figure(figsize=(3,9))
    #plt.subplot(raw,column,i)
    plt.axis('off')
    plt.tight_layout()
    #plt.figure()
    plt.imshow(img, interpolation='bilinear')
    plt.show()

#def printingAll(orgImg, nomarlizedImg, score_text, score_link):
    #printing(orgImg)
    #printing(nomarlizedImg)
    #printing(score_text)
    #printing(score_link)
#    plt.figure()
#    plt.imshow(orgImg,interpolation='bilinear')
#    plt.imshow(normarlizedImg,interpolation='bilinear')
#    plt.imshow(score_text,interpolation='bilinear')
#    plt.imshow(score_link,interpolation-'bilinear')
 #   
 #   plt.subplot(221)
  #  plt.subplot(222)
   # plt.subplot(212)

    #plt.show()

def getImgSize(img):
    print("IMG: {} HEIGHT: {}, WIDTH: {}, CHANNEL: {} LEN: {}".format(img.shape, img.shape[0], img.shape[1], img.shape[2], len(img.shape)))
    np.set_printoptions(threshold=sys.maxsize)
    #print("IMG HEIGHT 100th PIXEL VALUE --->\n{}".format(img[100]))

def getScalaValue(variable):
    print("Value: {}".format(variable))
