## How to use
1. run `python genImg.py` to generate the train images in `train/`, validation set in `test/`and the file name shall has the format of `00000001_name.png`, the number of process is set to `16`.
2. `cd standard` or `cd warpCTC`
3. run `python lstm_ocr.py` to training    

Notice that,  
- standard : use `tf.nn.ctc_loss` to calculate the ctc loss
- warpCTC : please install the [warpCTC tensorflow_binding](https://github.com/baidu-research/warp-ctc/tree/master/tensorflow_binding) first

### Dependency
- python 3  
- tensorflow 1.0.1  
- [captcha](https://pypi.python.org/pypi/captcha)
- (optional) [warpCTC tensorflow_binding](https://github.com/baidu-research/warp-ctc/tree/master/tensorflow_binding)

### Some details

The training data:  
![data](https://ooo.0o0.ooo/2017/04/13/58ef08ab6af03.png)  

Notice that, **sufficient amount of data is a must**, otherwise, the network cannot converge.  
optimizer can be found in `lstm_ocr.py`
Some tools and parameters can be found in `utils_*.py`  
some parameters need to be fined tune:
- learning rate
- decay step & decay rate (notice that, uncomment the learing rate decay part of code if it is commented)
- image_width
- image_height

if you want to use your own data and use pipline to read data, the height of the image shall be the same, besides, the suffix of the image shall be `png` or you can modify the code in `utils_*.py` from`tf.image.decode_png` to anything you need. However, if you read all your images at one time using OpenCV(default in my code), then this is not a problem.


### Result
update:
Notice that, different optimizer may lead to different resuilt.

---
After adding more training data
The accurary can be<del> more than 95%  
![accuracy](http://omy9d4djr.bkt.clouddn.com/markdown-img-paste-20170409223605283.png)  </del>
Read [this blog](https://ilovin.github.io/2017-04-06/tensorflow-lstm-ctc-ocr/) for more details and [this blog](http://ilovin.github.io/2017-04-23/tensorflow-lstm-ctc-input-output/) for how to
use `tf.nn.ctc_loss` or `warpCTC`
