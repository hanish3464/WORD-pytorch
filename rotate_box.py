import config
import codecs
import cv2
import numpy as np
theta = 78
def loadText(txt_file_path):
    a = []
    length = 0
    with codecs.open(txt_file_path, encoding='utf-8_sig') as file:
        coordinate = file.readlines()
        for line in coordinate:
            tmp = line.split(',')
            arr_coordinate = [int(n) for n in tmp]
            arr_coordinate = np.array(arr_coordinate).astype(float).reshape([4,2])
            arr_coordinate = arr_coordinate.tolist()
            a.append(arr_coordinate)
            length += 1
    return a, length

def rotate_bound(image, angle):
    h,w = image.shape[:2]
    cX, cY = w //2 , h//2
    M = cv2.getRotationMatrix2D((cX,cY),angle,1.0)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    nW = int((h*sin) + (w*cos))
    nH = int((h*cos) + (w*sin))
    
    M[0,2] += (nW/2) - cX
    M[1,2] += (nH/2) - cY
    return cv2.warpAffine(image,M,(nW,nH))

def rotate_box(bb, cx, cy, h, w):
    new_bb = list(bb)
    for i,coord in enumerate(bb):
        # opencv calculates standard transformation matrix
        M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
        # Grab  the rotation components of the matrix)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy
        #print(M)
        # Prepare the vector to be transformed
        v = [coord[0],coord[1],1]
        #print(v)
        # Perform the actual rotation and return the image
        calculated = np.dot(M,v)
        new_bb[i] = [calculated[0],calculated[1]]

    return new_bb

txt_path = config.json_gt_folder + '1.txt'

coordinate, length = loadText(txt_path)

# Original image
img_orig = cv2.imread(config.test_images_folder_path + '0001-003.jpg')
# Rotated image
rotated_img = rotate_bound(img_orig, theta)
rotated_img = cv2.resize(rotated_img, (1150, 450))
cv2.imwrite('./3.jpg', rotated_img)


# Calculate the shape of rotated images
(heigth, width) = img_orig.shape[:2]
print(heigth)
(cx, cy) = (width // 2, heigth // 2)
(new_height, new_width) = rotated_img.shape[:2]
(new_cx, new_cy) = (new_width // 2, new_height // 2)
print(cx,cy,new_cx,new_cy)
print(new_height, new_width)

## Calculate the new bounding box coordinates
new_bb = list()
#print(coordinate)
for i in range(length):
    print(coordinate[i])
    new_bb = rotate_box(coordinate[i], cx, cy, heigth, width)

    poly = np.array(new_bb).astype(np.int32).reshape((-1)).reshape(-1, 2)
    cv2.polylines(rotated_img, [poly.reshape((-1,1,2))], True, color=(255,0,0),thickness = 2)
    ptColor = (0,255,255)

cv2.imwrite('/home/hanish/workspace/temp_rot.jpg', rotated_img)


