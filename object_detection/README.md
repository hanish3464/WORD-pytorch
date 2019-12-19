# Object Detection in toon based on Faster R-CNN and OpenCV

```
The remaining issues : 
1. Result of bubble contour detection is not perfect. If bounding box of bubble is detected more elaborately with post processing, It is possible to improve performance.
2. Cut detection is a limit about the case to seperate cuts which obstacles(sound effects, characters, etc.) go through. So, if obstacle is firstly detached from cuts, It's also possible to improve performance.
```
### Sample Results

### Overview
This is Object Detection with Faster RCNN and Image Processing Technology. Firstly, faster RCNN detects bounging boxes of speech bubbles and then, segmentation result of speech bubble is detected by using image processing technologies
[(max area contours)](./bubble_utils.py). Secondly, Cut is detected from input image detached from speech bubble. I have used [numpy canvas, contours, and, convexHull methods](./cut_utils.py) to detect cut.

`cut results: [image source] : Dragon-Ball, Detective Conan, Naruto, One-Piece`
<img width="1200" height="500" src="./figures/bubble_demo.gif">

`bubble results: [image source] : Header of gangs, Free-throw, Zombie-Daughter, Cells-of-Yumi`
<img width="1200" height="500" src="./figures/cut_demo.gif">


## Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun
(Last revised on 6 Jan 2016(v3)) [Faster RCNN](https://arxiv.org/pdf/1506.01497.pdf)

## Train
Training code will be released in the near future. But, speech bubble datasets can't release bacause of [company](http://www.ideaconcert.com/) policy. I got all the data of speech bubbles from PSD file and suitable post processing method.

