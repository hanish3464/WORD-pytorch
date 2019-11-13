# Webtoon Speech Bubble Detection based on Faster R-CNN and OpenCV

`Note : This is Webtoon Speech Bubble Detector with Faster RCNN and OpenCV. It's not the final version code. I will the refine and update the code over and over again.`

## Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun
(Last revised on 6 Jan 2016(v3))

The full paper is available at: https://arxiv.org/pdf/1506.01497.pdf

## Install Requirements:
1、PyTroch=1.0.0(only)
```
pip install -r requirements.txt
```        
```
cd lib
python setup.py build develop
```    

## Training
`Note: When you train own your datasets, You must have Speech Bubble Bounding box labels. In other words, this code CAN segmentation Speech Bubble Object, but DON'T need segmetation data. pretrained model(Speech Bubble Detector) and training code will be released in the near future. But, speech bubble datasets can't release bacause of company policy`

## Test
`Note: When you test own your test images including speech bubble, You can get the segmentation data of speech bubble. below this`

`SAMPLE (image source: cells of Yumi, welcome to convinience store, love revolution, naver webtoon, and header of gangs)` 

<img src="https://github.com/hanish3464/webtoon_text_detection_with_CRAFT/blob/master/SPEECH_BUBBLE_DETECTION/test/images/sample1.jpg" width="192" height="384" /><img src="https://github.com/hanish3464/webtoon_text_detection_with_CRAFT/blob/master/SPEECH_BUBBLE_DETECTION/test/images/sample2.jpg" width="192" height="384" /><img src="https://github.com/hanish3464/webtoon_text_detection_with_CRAFT/blob/master/SPEECH_BUBBLE_DETECTION/test/images/sample3.jpg" width="192" height="384" /><img src="https://github.com/hanish3464/webtoon_text_detection_with_CRAFT/blob/master/SPEECH_BUBBLE_DETECTION/test/images/sample4.jpg" width="192" height="384" />

<img src="https://github.com/hanish3464/webtoon_text_detection_with_CRAFT/blob/master/SPEECH_BUBBLE_DETECTION/test/predictions/0.png" width="192" height="384" /><img src="https://github.com/hanish3464/webtoon_text_detection_with_CRAFT/blob/master/SPEECH_BUBBLE_DETECTION/test/predictions/1.png" width="192" height="384" /><img src="https://github.com/hanish3464/webtoon_text_detection_with_CRAFT/blob/master/SPEECH_BUBBLE_DETECTION/test/predictions/2.png" width="192" height="384" /><img src="https://github.com/hanish3464/webtoon_text_detection_with_CRAFT/blob/master/SPEECH_BUBBLE_DETECTION/test/predictions/3.png" width="192" height="384" />

- Run **`python kick_off.py --test 1`**
# Acknowledgement
Thanks for jwyang excellent work and code
https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0) for train and test. 
