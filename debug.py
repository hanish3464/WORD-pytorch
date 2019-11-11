'''debug.py'''

import matplotlib.pyplot as plt
import cv2

def checkImageOnGUI(img):
    h,w = img.shape
    cv2.resize(img, (w*2,h*2), interpolation = cv2.INTER_LINEAR)
    plt.figure(figsize=(3,9))
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(img, interpolation='bilinear')
    plt.show()

