import numpy as np

# MODEL PATH
SPEECH_BUBBLE_DETECTOR_PATH = './weights/Speech-Bubble-Detector.pth'
TEXT_DETECTOR_MODEL_PATH  = './weights/Line-Text-Detector.pth'
TEXT_RECOGNIZER_MODEL_PATH = './weights/Line-Text-Recognizer.pth'

# TRANSLATION
# You can get the ID and P/W to https://developers.naver.com/products/nmt/
PAPAGO_ID = 'id'
PAPAGO_PW = 'pw'

# CUDA
cuda = True

# BUBBLE DETECTION
LABEL = np.asarray(['__background__', 'speech'])
RNG_SEED = 3
DRAWTXT = False
DRAWLINK = False
DRAWBUB = False
DRAWCUT = True
BUBBLE_TRAIN_FLIP = False
BACKBONE = 'res101'

# TEXT DETECTION
gaussian_region = 0.3
gaussian_affinity = 0.25
rotate = False
flip = False
crop = False
DETECTION_TRAIN_IMAGE_PATH = './train/images/'
DETECTION_TRAIN_LABEL_PATH = './train/labels/'
text_threshold = 0.6
low_text = 0.4
link_threshold = 0.4
MARGIN = 3
LEFT_CHAR_LINE = 0
RIGHT_CHAR_LINE = 2

# TEXT RECOGNITION
RECOG_NET_CHANNEL = 1
RECOG_TRAIN_SIZE = 224
NUM_CLASSES = 2213
WARP_SPACE_THRESHOLD = 10
RECOGNITIOON_FONT_PATH = './train/fonts/'
RECOGNITION_CSV_PATH = './train/label.csv'
RECOGNITION_TRAIN_IMAGE_PATH = './train/images/'
RECOG_IMAGE_WIDTH = RECOG_IMAGE_HEIGHT = 64
RECOG_BACKGROUND = 0 # black
RECOG_FONT_COLOR = 255
MORPH_NUM = 2  # the number of data augmentation morphologyEx : dilate, erode
RECOG_FONT_SIZE = 48
RECOG_WEBTOON_TRAIN_DATA_PATH = './train/webtoon/data/'
RECOG_WEBTOON_TRAIN_LABEL_PATH = './train/webtoon/labels.txt'

# CUT IMAGE AWAY
PIXEL_THRESHOLD = 500
MIN_SIZE = 100


# TEST HYPER PARAMETER
TEST_SCALES = (600,)
TEST_MAX_SIZE = 1000
PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
TEST_BBOX_REG = True
TEST_NMS = 0.3

BACKGROUND = 'white'
OPENING_ITER_NUM = 1
DILATE_ITER_NUM = 1
THRESH_EXTENT = 127
KERNEL_SIZE = 25

# TRAIN HYPER PARAMETER
TRAIN_BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
TRAIN_BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)
TRAIN_BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True

# BBOX THRESHOLD
THRESH = 0.05
BBOX_BORDER_THRESH = 6
char_box_width_threshold = -5
char_box_height_threshold = 3

# test
target_size = 1024
white = [255, 255, 255]
recognition_input_size = 224
LNK_KERNEL_SIZE = 50

MAG_RATIO = 2.5
MAXIMUM_IMAGE_SIZE = 4000
TRAIN_IMAGE_SIZE = 512



