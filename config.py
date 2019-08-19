train_img_num = 16
img_num = 16
gt_json_num = 18
gt_xml_num = 500

#convertor path
json_gt_folder = './conversion/json_to_txt_gt/'
json_path = './conversion/json_gt/'

#pretrained path
prediction_folder = './original/prediction/'
test_images = './original/test_images/'
ground_truth = './original/ground_truth/'
mask_folder = './original/mask/'
ground_truth_folder = './original/ground_truth/res_'

#model
pretrained_model_path = './model/pretrained/clova_ai_CRAFT.pth'
saving_model = './model/train_model/'

#test path
test_ground_truth = './test/test_ground_truth/'
test_images_folder_path = './test/test_images/'
test_prediction_image = './test/test_prediction_image/'
test_prediction_folder = './test/test_prediction/'

#train path
train_ground_truth_folder = './train/train_ground_truth/'
train_images_folder_path = './train/train_images/'
train_prediction_folder = './train/train_prediction/'

#threshold
iou_threshold = 0.4
text_threshold = 0.7
low_text = 0.4
link_threshold = 0.4
cuda = True
mag_ratio = 1.5
image_size = 4000
poly = False
show_time = False
num_of_gpu = 1
data_augmentation_rotate = True

data_augmentation_crop = True
pos_crop_bound_threshold = 5
neg_crop_bound_threshold = -5

data_augmentation_flip = True


#parameter
learning_rate = 1e-4
lr_multiply =0.8 #10K iterations
batch_size = 4 #8 batch per 1 GPU
epoch = 100000 #50K about synthText 70K
iterations = 20

