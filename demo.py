"""
This project name is WORD(Webtoon Object Recognition and Detection)
WORD consists of object detection (detection of speech bubble, cut) and OCR(detection and recognition of line text)
Yon can also meet results of translation with papago API of naver corp if you want.


Future Science Technology Internship
Ajou University.
Writer: Han Kim


referenced paper :
            Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
            CRAFT : Character Region Awareness for Text Detection
"""

import argparse
from cut_off import cut_off_image as cut_off
from object_detection.bubble import test_net as bubble_detect
from object_detection.cut import test_opencv as cut_detect
from text_detection.line_text import test as line_text_detect
from text_recognition.line_text import test_net as line_text_recognize
from text_recognition.ltr_utils import gen_txt_to_image as gen_text_to_image
from translation.papago import translation as papago_translation
import file_utils
import net_utils
import imgproc
import time
import opt

parser = argparse.ArgumentParser(description='WORD(Webtoon Object Recognition and Detection')

parser.add_argument('--object_detector', default='./weights/Speech-Bubble-Detector.pth', type=str, help='pretrained')
parser.add_argument('--text_detector', default='./weights/Line-Text-Detector.pth', type=str, help='pretrained')
parser.add_argument('--text_recognizer', default='./weights/Line-Text-Recognizer.pth', type=str, help='pretrained')
parser.add_argument('--object', action='store_true', default=True, help='enable objext detection')
parser.add_argument('--ocr', action='store_true', default=False, help='enable OCR')
parser.add_argument('--papago', action='store_true', default=False, help='enable English translation with papago')
parser.add_argument('--type', default='white', type=str, help='background type: white, black, classic')
parser.add_argument('--cls', default=0.995, type=float, help='bubble prediction threshold')
parser.add_argument('--box_size', default=7000, type=int, help='cut size filtering threshold')
parser.add_argument('--large_scale', action='store_true', default=False, help='demo image is large scale')
parser.add_argument('--ratio', default=2.0, type=float, help='height & width ratio of demo image')
parser.add_argument('--demo_folder', default='./data/', type=str, help='folder path to demo images')
parser.add_argument('--cuda', action='store_true', default=True, help='use cuda for inference')
parser.add_argument('--cpu', action='store_true', default=False, help='use CPU for inference')

# Add a new command line argument for the result path
parser.add_argument('--result', default='./result/', type=str, help='folder path to save results')

# Parse the arguments on the command line. The result is stored in the args variable
args = parser.parse_args()

# If --cpu is specified, disable CUDA
if args.cpu:
    args.cuda = False

# Use the result path from the command line arguments
result_path = args.result

# Make sure the result path ends with a slash
if not result_path.endswith('/'):
    result_path += '/'

""" For test images in a folder """
image_list, _, _, name_list = file_utils.get_files(args.demo_folder)

# Replace all occurrences of './result/' with the result path
file_utils.rm_all_dir(dir=result_path)  # clean directories for next test
file_utils.mkdir(dir=[result_path, result_path+'bubbles/', result_path+'cuts/', result_path+'demo/', result_path+'chars/'])

# load net
models = net_utils.load_net(args)  # initialize and load weights

spaces = []  # text recognition spacing word
text_warp_items = []  # text to warp bubble image
demos = []  # all demo image storage
t = time.time()

cnt = 0

# load data
for k, image_path in enumerate(image_list):
    
    # Print the test image information
    print("TEST IMAGE ({:d}/{:d}): INPUT PATH:[{:s}]".format(k + 1, len(image_list), image_path), end='\n')

    # Load the image
    img = imgproc.loadImage(image_path)

    # Check if the image is large scale
    if args.large_scale:
        # Cut off large scale images into several pieces (width : height = 1 : 2)
        images = cut_off(image=img, name=name_list[k], ratio=args.ratio)
    else:
        # Uniformize the shape of the image for general scale case
        images = imgproc.uniformizeShape(image=img)

    for img in images:  # image fragments divided from cut_off.py

        # Increment the counter
        cnt += 1

        # Convert the counter to a string with leading zeros
        str_cnt = file_utils.resultNameNumbering(origin=cnt, digit=1000)  # ex: 1 -> 0001, 2 -> 0002

        # Get the image blob and scale
        img_blob, img_scale = imgproc.getImageBlob(img)
        f_RCNN_param = [img_blob, img_scale, opt.LABEL]  # LABEL: speech bubble

        # Perform bubble detection on the image
        demo, image, bubbles, dets_bubbles = bubble_detect(model=models['bubble_detector'], image=img,
                                                           params=f_RCNN_param, cls=args.cls, bg=args.type)

        # Perform cut detection on the image
        demo, cuts = cut_detect(image=image, demo=demo, bg=args.type, size=args.box_size)

        # Perform line text detection on the image
        demo, space, warps = line_text_detect(model=models['text_detector'], demo=demo,
                                              bubbles=imgproc.cpImage(bubbles),
                                              dets=dets_bubbles, img_name=str_cnt, save_to='./result/chars/')

        # Add the temporary spaces in the image to the total spaces storage
        spaces += space

        # Add the temporary text and bubble images to the text_warp_items storage
        text_warp_items += warps

        # Add the demo image to the demos storage
        demos.append(demo)

        # Save all the bubble images
        file_utils.saveAllImages(save_to=result_path+'bubbles/', imgs=bubbles, index1=str_cnt, ext='.png')

        # Save all the cut images
        file_utils.saveAllImages(save_to=result_path+'cuts/', imgs=cuts, index1=str_cnt, ext='.png')


if args.ocr:  # ocr

    # save spaces word information
    file_utils.saveText(save_to=result_path, text=spaces, name='spaces')


    # mapping one-hot-vectors to hangul labels
    label_mapper = file_utils.makeLabelMapper(load_from='./text_recognition/labels-2213.txt')

    # load spacing word information
    spaces, _ = file_utils.loadSpacingWordInfo(load_from=result_path+'spaces.txt')


    # Measure the starting time
    x = time.time()

    # Print a message indicating that OCR processing is in progress
    print('\n[processing ocr.. please wait..]', end=' ')

    # Perform line text recognition using the specified model, label mapper, spaces, and input/output paths
    line_text_recognize(model=models['text_recognizer'], mapper=label_mapper, spaces=spaces,
                        load_from=result_path+'chars/', save_to=result_path+'ocr.txt')

    # Calculate and print the elapsed time for OCR processing
    print("[ocr time: {:.6f} sec]".format(time.time() - x))

# Perform translation from Korean to English if the 'papago' flag is set
if args.papago:
    print('[translating korean to English..]')
    papago_translation(load_from=result_path+'ocr.txt', save_to=result_path+'english_ocr.txt', id=opt.PAPAGO_ID, pw=opt.PAPAGO_PW)

# Generate text-to-image using the translated English OCR results
gen_text_to_image(load_from='./result/english_ocr.txt', warp_item=text_warp_items)

# Save all demo images
print("Saving all demo images")
file_utils.saveAllImages(save_to=result_path+'demo/', imgs=demos, ext='.png')

# Calculate and print the elapsed time
print("[elapsed time : {:.6f} sec]".format(time.time() - t))
