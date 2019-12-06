#MODEL PATH
#PRETRAINED_MODEL_PATH = './saved_models/clova_ai_CRAFT.pth'
PRETRAINED_MODEL_PATH = './saved_models/wtd400.pth'
REFINER_MODEL_PATH = './pretrained_models/craft_refiner_CTW1500.pth'
SAVED_MODEL_PATH = './saved_models/'

#original path
orig_ground_truth = "./original/ground_truth/"
orig_img = './original/test_images/'
orig_pdt_img = './original/prediction/'
orig_mask = './original/mask/'

#TEST PATH
TEST_IMAGE_PATH = '../object_detection/test/predictions/bubble/'
TEST_PREDICTION_PATH = './test/predictions/'
CANVAS_PATH = TEST_PREDICTION_PATH + 'canvas/'
MASK_PATH = TEST_PREDICTION_PATH + 'mask/'
BBOX_PATH = TEST_PREDICTION_PATH + 'bbox/'
RESULT_CHAR_PATH = TEST_PREDICTION_PATH + 'res/'
SPACING_WORD_PATH = TEST_PREDICTION_PATH + 'spacing_word/'

#TRAIN PATH
TRAIN_LABEL_PATH = './train/labels/'
TRAIN_IMAGE_PATH = './train/images/'
TRAIN_PREDICTION_PATH = './train/predictions/'

#PSD path
jpg_images_folder_path = './psd/jpg_images/'
jpg_cropped_images_folder_path = './psd/cropped_images/'
jpg_text_ground_truth= './psd/text_ground_truth/'


#threshold
iou_threshold = 0.4
text_threshold = 0.6
low_text = 0.4
link_threshold = 0.4

divide_text_threshold = 0.7
divide_low_text = 0.4
divide_link_threshold = 0.4

char_box_width_threshold = -5
char_box_height_threshold = 3

#test
target_size = 1024
white = [255, 255, 255]
recognition_input_size = 224
LNK_KERNEL_SIZE = 50

cuda = True
MAG_RATIO = 1.5
MAXIMUM_IMAGE_SIZE = 4000
TRAIN_IMAGE_SIZE = 512
poly = False
show_time = False
num_of_gpu = 1
data_augmentation_rotate = True

data_augmentation_crop = True
pos_crop_bound_threshold = 5
neg_crop_bound_threshold = -5

pos_link_threshold = 0
neg_link_threshold = -5

data_augmentation_flip = True

gaussian_region = 0.3
gaussian_affinity = 0.25

#parameter
LEARNING_RATE = 1e-4
LR_DECAY_STEP = 5
WEIGHT_DECAY = 5e-4
lr_multiply =0.8 #10K iterations
BATCH = 1
EPOCH = 500

VIS = True
DISPLAY_INTERVAL = 20
SAVE_INTERVAL = 100
