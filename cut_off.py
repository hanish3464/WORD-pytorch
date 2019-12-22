import cv2
import opt
import imgproc

storage = []


def search_pixels(img, h_criteria):
    h, w, _ = img.shape
    arr = []
    arr.append(h_criteria)
    mag = 1
    while arr:
        h_c = arr.pop(0)
        if h_c < 0 or h_c >= h:
            print('[error]Image out of index')
            return 'err'
        if h_c < h_criteria - opt.PIXEL_THRESHOLD or h_c >= h_criteria + opt.PIXEL_THRESHOLD:
            print('It takes too long times. so, cut down image as initial criteria')
            return h_criteria

        if (img[h_c] == 0).all() or (img[h_c] == 255).all(): return h_c
        arr.append(h_criteria + mag)
        arr.append(h_criteria - mag)
        mag += 1


def down_size_image(img, width=None):
    h, w, _ = img.shape
    ratio_h = int((width * h) // w)
    criteria_w = w / width
    img = cv2.resize(img, (width, ratio_h), interpolation=cv2.INTER_CUBIC)
    return img, criteria_w


def bfs(origin=None, copy=None, criteria_h=None, creteria_w=None, name=None, index=None):
    global storage
    h, _, _ = copy.shape
    if h <= criteria_h:
        piece = origin[0:int(h * creteria_w), :, :]
        if h >= opt.MIN_SIZE:
            storage.append(piece)
        return

    n_h = search_pixels(copy, criteria_h)
    piece = origin[0:int(n_h * creteria_w), :, :]
    storage.append(piece)

    bfs(origin=origin[int(n_h * creteria_w):, :, :], copy=copy[n_h:, :, :], criteria_h=criteria_h, creteria_w=creteria_w,
        name=name, index=index + 1)


def cut_off_image(image=None, name=None, ratio=None):
    global storage
    storage = []
    if image.shape[1] >= 720:
        h, w, _ = image.shape
        ratio_h = int((720 * h) // w)
        image = cv2.resize(image, (720, ratio_h), interpolation=cv2.INTER_CUBIC)

    copy = imgproc.cpImage(img=image)
    copy, criteria_w = down_size_image(copy, width=500)  # Resize image for searching pixels to decrease times.
    height, width, _ = image.shape
    copy_height, copy_width, _ = copy.shape
    criteria_h = int(copy_width * ratio)
    bfs(origin=image, copy=copy, criteria_h=criteria_h, creteria_w=criteria_w, name=name, index=0)

    return storage
