- [old master](https://github.com/ilovin/lstm_ctc_ocr/tree/backup):
    - harder to converge compare to the beta version
    - both standard ctc and warpCTC
    - read data at once
- [dev](https://github.com/ilovin/lstm_ctc_ocr/tree/dev):
    - the pipline version of lstm_ctc_ocr, resize to same size
    - use tf.records
- [beta](https://github.com/ilovin/lstm_ctc_ocr/tree/beta) (current):
    - generate data on the fly
    - deal with multi-width image, padding to same width

## How to use
1. ./train.sh


### Dependency
- python 3
- tensorflow 1.0.1  
- [captcha](https://pypi.python.org/pypi/captcha)
- [warpCTC tensorflow_binding](https://github.com/baidu-research/warp-ctc/tree/master/tensorflow_binding)

### Some details

The training data:  
![data](https://ooo.0o0.ooo/2017/04/13/58ef08ab6af03.png)  

Notice that,
parameters can be found in `./lstm.yml`(higher priority) and `lib/lstm/utils/config.y`  
some parameters need to be fined tune:
- learning rate
- decay step & decay rate
- image_height
- optimizer?

in `./lib/lstm/utils/gen.py`, the height of the images are the same, and I pad the width
to the same for each batch, so
if you want to use your own data, the height of the image shall be the same.

### Result
The accurary can be more that 95%
![acc](https://i.loli.net/2017/08/28/59a2ee75a2a0a.png)

Read [this blog](https://ilovin.github.io/2017-04-06/tensorflow-lstm-ctc-ocr/) for more details and [this blog](http://ilovin.github.io/2017-04-23/tensorflow-lstm-ctc-input-output/) for how to
use `tf.nn.ctc_loss` or `warpCTC`
