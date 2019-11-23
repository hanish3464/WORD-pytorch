# Webtoon Cut & Speech Bubble Detection based on Faster R-CNN and OpenCV

`Note : This is Webtoon Cut & Speech Bubble Detection with Faster RCNN and Image Processing Technology. It's not the final version code. I will the refine and update the code over and over again.`

## Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun
(Last revised on 6 Jan 2016(v3)) [Faster RCNN](https://arxiv.org/pdf/1506.01497.pdf)

# Update 


1、2019.11.12(Tue) : Add Speech Bubble Detection Test Results.

2、2019.11.21(Thu) : Add Webtoon Cut Detection Test Results and Pretrained Model

## Install Requirements:
1、Pytorch==1.0.0(only)
```
pip install -r requirements.txt
```        
```
cd lib
python setup.py build develop
```    

## Pretrained Models
 *Model name* | *Model Link* |
 | :--- | :--- |
Speech Bubble Detector | [Click](https://drive.google.com/open?id=1F10sRXWuICKuSQclaUnQVBo1rlxa6ogR)


`Download model and include pretrained_models/`


## Training
`Note: When you train own your datasets, You must have Speech Bubble Bounding box labels. In other words, this code CAN segmentation Speech Bubble Object, but DON'T need segmetation data. pretrained model(Speech Bubble Detector) and training code will be released in the near future. But, speech bubble datasets can't release bacause of company policy`

## Test
`Note: When you test own your test images including speech bubble, You can get the segmentation data of speech bubble. below this`

`SAMPLE (image source: cells of Yumi, welcome to convinience store, love revolution, naver webtoon, and header of gangs)` 

<img src="https://github.com/hanish34/ideaconcert/blob/master/SPEECH_BUBBLE_DETECTION/sample/images/sample1.jpg" width="192" height="768" /><img src="https://github.com/hanish34/ideaconcert/blob/master/SPEECH_BUBBLE_DETECTION/sample/images/sample2.jpg" width="192" height="768" /><img src="https://github.com/hanish34/ideaconcert/blob/master/SPEECH_BUBBLE_DETECTION/sample/images/sample3.jpg" width="192" height="768" /><img src="https://github.com/hanish34/ideaconcert/blob/master/SPEECH_BUBBLE_DETECTION/sample/images/sample4.jpg" width="192" height="768" />


<img src="https://github.com/hanish34/ideaconcert/blob/master/SPEECH_BUBBLE_DETECTION/sample/res/0.jpg" width="192" height="768" /><img src="https://github.com/hanish34/ideaconcert/blob/master/SPEECH_BUBBLE_DETECTION/sample/res/1.jpg" width="192" height="768" /><img src="https://github.com/hanish34/ideaconcert/blob/master/SPEECH_BUBBLE_DETECTION/sample/res/2.jpg" width="192" height="768" /><img src="https://github.com/hanish34/ideaconcert/blob/master/SPEECH_BUBBLE_DETECTION/sample/res/3.jpg" width="192" height="768" />

- Run **`python kick_off.py --test 1`**
# Acknowledgement
Thanks for jwyang excellent work and code
https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0) for train and test. 
