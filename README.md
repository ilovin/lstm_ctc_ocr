## How to use
1. run `python genImg.py` to generate the train images in `train/`, validation set in `test/`and the file name shall has the format of `00000001_name.png`
2. run `python lstm_ocr.py` to training  

### Dependency
- python 3  
- tensorflow 1.0.1  
- [captcha](https://pypi.python.org/pypi/captcha)

![](http://omy9d4djr.bkt.clouddn.com/markdown-img-paste-20170407164955997.png)  

Notice that, Some tools and parameters can be found in `utils.py`  
The accurary is more than 95%  
![accuracy](http://omy9d4djr.bkt.clouddn.com/markdown-img-paste-20170409223605283.png)
Read [this blog](https://ilovin.github.io/2017-04-06/tensorflow-lstm-ctc-ocr/) for more detail
