train_img_num =2 
img_num = 3
gt_xml_num = 500

#path
json_gt_folder = './json_to_txt_gt/label_'
json_path = './json_gt/'
prediction_folder = './prediction/'
mask_folder = './mask/'
ground_truth_folder = './ground_truth/res_'
train_ground_truth_folder = './train_ground_truth/'
pretrained_model_path = '/home/hanish/workspace/clova_ai_CRAFT.pth'
test_images_folder_path = '/home/hanish/workspace/test_images'
train_images_folder_path = '/home/hanish/workspace/train_images/'
train_prediction_folder = './train_prediction/'
image_size = 3000

#parser.add_argument('--pretrained_model_path', default='/home/hanish/workspace/clova_ai_CRAFT.pth', type=str, help='pretrained model')
#parser.add_argument('--test_images_folder_path', default='/home/hanish/workspace/test_images',type=str, help='path to test_input images')
#parser.add_argument('--image_size', default=3000, type=int, help='image size')

#threshold
iou_threshold = 0.5
text_threshold = 0.7
low_text = 0.4
link_threshold = 0.4
cuda = True
mag_ratio = 1.5
poly = False
show_time = False
num_of_gpu = 1

#parameter
learning_rate = 1e-4
lr_multiply =0.8 #10K iterations
batch_size = 1 #8 batch per 1 GPU
epoch = 100000 #50K about synthText 70K
iterations = 20


#parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
#parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
#parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
#parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
#parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
#parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
#parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
