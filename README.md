1. [master](https://github.com/ilovin/lstm_ctc_ocr/tree/master):  
  - both standard ctc and warpCTC
  - read data at one time
2. [dev](https://github.com/ilovin/lstm_ctc_ocr/tree/dev)(current):  
  - the pipline version of lstm_ctc_ocr, resize to same size
3. [beta](https://github.com/ilovin/lstm_ctc_ocr/tree/beta):  
  - generate data on the fly(highest accuracy)
  - deal with multi-width image, padding to same width

## How to use
1. run `python ./lib/utils/genImg.py` to generate the train images in `train/`, validation set in `val`and the file name shall has the format of `00000001_name.png`, the number of process is set to `16`.
2. `python ./lib/lstm/utils/tf_records.py` to generate tf_records file, which includes both images and labels(the `img_path` shall be changed to your image_path)
3. `./train.sh` for training `./test.sh`for testing

Notice that,  
the pipline version use warpCTC as default : please install the [warpCTC tensorflow_binding](https://github.com/baidu-research/warp-ctc/tree/master/tensorflow_binding) first  
if your machine does not support warpCTC, then use `standard` ctc version in [the master branch](https://github.com/ilovin/lstm_ctc_ocr/tree/master)
- standard CTC: use `tf.nn.ctc_loss` to calculate the ctc loss

### Dependency
- python 3  
- tensorflow 1.0.1  
- [captcha](https://pypi.python.org/pypi/captcha)
- [warpCTC tensorflow_binding](https://github.com/baidu-research/warp-ctc/tree/master/tensorflow_binding)

### Some details

The training data:  
![data](https://ooo.0o0.ooo/2017/04/13/58ef08ab6af03.png)  

Notice that, **sufficient amount of data is a must**, otherwise, the network cannot converge.  
parameters can be found in `./lstm.yml`(higher priority) and `lib/lstm/utils`  
some parameters need to be fined tune:
- learning rate
- decay step & decay rate
- image_width
- image_height
- optimizer?

in `./lib/lstm/utils/tf_records.py`, I resize the images to the same size.
if you want to use your own data and use pipline to read data, the height of the image shall be the same.

### Result
update:
Notice that, different optimizer may lead to different resuilt.

---
The accurary is about 85%~92% (training on 128k images)
![acc](https://i.loli.net/2017/07/17/596c6de6584f7.png)

Read [this blog](https://ilovin.github.io/2017-04-06/tensorflow-lstm-ctc-ocr/) for more details and [this blog](http://ilovin.github.io/2017-04-23/tensorflow-lstm-ctc-input-output/) for how to
use `tf.nn.ctc_loss` or `warpCTC`
