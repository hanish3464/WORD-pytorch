# Webtoon Text Detection based on CRAFT(Character-Region Awareness For Text detection)

`Note : This is Webtoon Text(character) Detector with OpenCV and CRAFT. It's not the final version code. I will the refine and update the code over and over again.`

## Character Region Awareness for Text Detection
Youngmin Baek, Bado Lee, Dongyoon Han, Sangdoo Yun, Hwalsuk Lee
(Submitted on 3 Apr 2019)

The full paper is available at: https://arxiv.org/pdf/1904.01941.pdf                                                         

## Install Requirements:                                                                                                        
1、PyTroch>=0.4.1                                                                                                                             
2、torchvision>=0.2.1 			                                                    																			                             
3、opencv-python>=3.4.2    
```
pip install -r requirements.txt
```                                                                                                                                                                                
## Training 
`Note: When you train own your datasets, You must have character Anotations like SynthText. In other words, this code doesn't          contain weakly supervision part. We don't need this part because webtoon image is synthetic(possible to get character datasets by generating characters like synthText. But, character dataset can't release because of company policy.`                                         
- Run **`python kick_off.py --train 1`**

## Test
`Note: When you test own your test images, You can get the characters on images for recognition`

- Run **`python kick_off.py --test 1`**

`SAMPLE (image source: Welcome to Convenience Store, naver webtoon)` 

<img src="https://github.com/hanish3464/webtoon_text_detection_with_CRAFT/blob/master/WEBTOON_TEXT_DETECTION/sample/sample1.png" width="384" height="256" /><img src="https://github.com/hanish3464/webtoon_text_detection_with_CRAFT/blob/master/WEBTOON_TEXT_DETECTION/sample/sample2.png" width="384" height="256" />

`final result` 

<img src="https://github.com/hanish3464/webtoon_text_detection_with_CRAFT/blob/master/WEBTOON_TEXT_DETECTION/sample/sample1_res.jpg" width="384" height="128" />

## Evaluation
`Note: Evalution is under the developing. so, It can not run the code now. I will use IOU Method and update as soon as possible.`

- Run **`python kick_off.py --evaluation 1`**
                                                    

# Acknowledgement
Thanks for Youngmin Baek, Bado Lee, Dongyoon Han, Sangdoo Yun, Hwalsuk Lee excellent work and [code](https://github.com/clovaai/CRAFT-pytorch) for test. In this repo, we use the author repo's basenet(backbone)
