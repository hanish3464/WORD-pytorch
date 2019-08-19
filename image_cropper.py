import cv2
import numpy as np
import preprocess
import file

img_list = file.get_files('./temp/')[0]
print(img_list)
print(len(img_list))

def crop(img, lens):

	height, width, channel = img.shape
	print(img.shape)
	np_img= np.array(img)
	if width >= height:
		distance = width-height /2
		temp = width
		
		canvas = np.ones((temp, width, channel))

	else:
		temp = height
		distance = height - width
		canvas = np.ones((height, temp, channel))
	
	h, w, _ = canvas.shape
	canvas[0:height, 0:width , ::-1] = np_img
	cv2.imwrite('./temp/resize_img/' + str(lens) + '.jpg', canvas)

lens = 1
for i in img_list:
	img = preprocess.loadImage(i)
	crop(img, lens)
	lens += 1
	