import cv2
import numpy as np
import preprocess

x_min, y_min, x_max, y_max = 579, 696, 922, 920
temp_img = 'psd_test.jpg'
img = preprocess.loadImage(temp_img)
np.array(img)
poly = np.array([[x_min,y_min], [x_max,y_min], [x_max,y_max], [x_min,y_max]])
#cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(255, 0, 0), thickness=3)
#cv2.imwrite('psd_draw_box_img.jpg', img)
cropped_img = img[y_min:y_max, x_min:x_max]
cv2.imwrite('psd_cropped_img.jpg', cropped_img)
