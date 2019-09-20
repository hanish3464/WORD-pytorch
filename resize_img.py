import cv2
import preprocess
path = "./1.jpg"
img = preprocess.loadImage(path)
img=cv2.resize(img,(512,512), interpolation=cv2.INTER_LINEAR)
cv2.imwrite('./tmp8.jpg', img[:,:,::-1])
