train_img_num = 7
img_num = 16
gt_json_num = 21
gt_xml_num = 500

#convertor path
json_gt_folder = './conversion/json_to_txt_gt/'
json_path = './conversion/json_gt/'

#model
pretrained_model_path = './model/pretrained/clova_ai_CRAFT.pth'
saving_model = './model/train_model/'

#original path
orig_ground_truth = "./original/ground_truth/"
orig_img = './original/test_images/'
orig_pdt_img = './original/prediction/'
orig_mask = './original/mask/'

#test path
test_ground_truth = './test/test_ground_truth/'
test_images_folder_path = './test/test_images/'
test_prediction_image = './test/test_prediction_image/'
test_mask = './test/test_mask/'

#train path
train_char_gt_path = './train/train_char_gt_path/'
train_images_path = './train/train_images/'
train_prediction_path = './train/train_prediction/'
train_word_gt_path = './train/train_word_gt_path/'

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

cuda = True
mag_ratio = 1.5
char_annotation_cropped_img_ratio = 2.5
image_size = 4000
train_image_size = 512
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

gaussian_spread = 1.2
gaussian_sigma = 10

#parameter
lr = 1e-4
weight_decay = 5e-4
lr_multiply =0.8 #10K iterations
batch = 4 #8 batch per 1 GPU
epoch = 100000 #50K about synthText 70K
iterations = 20

