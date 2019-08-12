import config
import codecs
import cv2
import numpy as np
import debug

def loadText(txt_file_path):
    with codecs.open(txt_file_path, encoding='utf-8_sig') as file:
        a = list()
        while True:
            coordinate = file.readlines()
            if coordinate is None:
                break
            for line in coordinate:
                tmp = line.split(',')
                arr_coordinate = [int(n) for n in tmp]
                coordinate = np.array(arr_coordinate).astype(float)
                coordinate = coordinate.tolist()
                a = a.append(coordinate)
    return a

def rotate_bound(image, angle):
    h,w = image.shape[:2]
    cX, cY = w //2 , h//2
    M = cv2.getRotationMatrix2D((cX,cY),angle,1.0)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,0])
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
        # Prepare the vector to be transformed
        v = [coord[0],coord[1],1]
        # Perform the actual rotation and return the image
        calculated = np.dot(M,v)
        new_bb[i] = (calculated[0],calculated[1])
    return new_bb

txt_path = config.json_gt_folder + '1.txt'

coordinate = loadText(txt_path)
coordinate  = np.array(coordinate)


print(coordinate)

# Original image
img_orig = cv2.imread(config.test_images_folder_path + '0001-001.jpg')
# Rotated image
rotated_img = rotate_bound(img_orig, 30)
debug.printing(rotated_img)

# Calculate the shape of rotated images
(heigth, width) = img_orig.shape[:2]
(cx, cy) = (width // 2, heigth // 2)
(new_height, new_width) = rotated_img.shape[:2]
(new_cx, new_cy) = (new_width // 2, new_height // 2)
print(cx,cy,new_cx,new_cy)
print(new_height, new_width)

## Calculate the new bounding box coordinates
new_bb = {}
for i in range(8):
    new_bb[i] = rotate_box(coordinate[i], cx, cy, heigth, width)

## Plot rotated image and bounding boxes
ax2.imshow(rotated_img[...,::-1], aspect='auto')
ax2.axis('off')
ax2.add_patch(mpatches.Polygon(new_bb[0],lw=3.0, fill=False, color='red'))
ax2.add_patch(mpatches.Polygon(new_bb[1],lw=3.0, fill=False, color='red'))
ax2.add_patch(mpatches.Polygon(new_bb[2],lw=3.0, fill=False, color='green'))
ax2.text(0.,0.,'Rotation by: ' + str(theta), transform=ax1.transAxes,
           horizontalalignment='left', verticalalignment='bottom', fontsize=30)
name='Output.png'
plt.savefig(name)
plt.cla()
