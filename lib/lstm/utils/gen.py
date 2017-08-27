# encoding:utf-8
# encoding:utf-8
import glob
import csv
import cv2,math
import time
import os,random
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.patches as Patches
#from shapely.geometry import Polygon
from lib.utils.data_util import GeneratorEnqueuer
from lib.lstm.config import cfg,get_encode_decode_dict
import tensorflow as tf
from captcha.image import ImageCaptcha
import cv2
import sys


def randRGB():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def gen_rand():
    buf = ""
    max_len = random.randint(cfg.MIN_LEN,cfg.MAX_LEN)
    for i in range(max_len):
        buf += random.choice(cfg.CHARSET)
    return buf

def generateImg():
    captcha=ImageCaptcha(fonts=[cfg.FONT])
    if not os.path.exists(cfg.FONT):
        print('cannot open the font')
    theChars=gen_rand()
    data=captcha.generate_image(theChars)
    return np.array(data),theChars

encode_maps,decode_maps = get_encode_decode_dict()

def groupBatch(imgs,labels):
    max_w = -sys.maxsize
    time_steps = []
    label_len = []
    label_vec = []
    img_batch = []
    for i,img in enumerate(imgs):
        if cfg.NCHANNELS==1: h,w = img.shape
        else: h,w,_ = img.shape
        max_w = max(max_w,w)
        time_steps.append(w//cfg.POOL_SCALE)
        code = [encode_maps[c] for c in list(labels[i])]
        label_vec.extend(code)
        label_len.append(len(labels[i]))
    max_w = math.ceil(max_w/cfg.POOL_SCALE)*cfg.POOL_SCALE
    for img in imgs:
        if cfg.NCHANNELS==1: h,w = img.shape
        else: h,w,_ = img.shape
        img = cv2.copyMakeBorder(img,0,0,0,max_w-w,cv2.BORDER_CONSTANT,value=0).astype(np.float32)/255.
        img = img.swapaxes(0, 1)
        img = np.reshape(img,[-1,cfg.NUM_FEATURES])
        img_batch.append(img)
    #img_batch = np.array(img_batch)
    return img_batch,label_vec,label_len,time_steps

def generator(batch_size=32, vis=False):
    images = []
    labels = []
    while True:
        try:
            im, label = generateImg()
            #img_size = cfg.IMG_SHAPE  # 160,60
            #im = cv2.resize(im,(img_size[0],img_size[1]))
            if cfg.NCHANNELS == 1:
                im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
                #im = np.expand_dims(im,axis=2)
            # print(im.shape,' ',label)
            if vis:
                fig, axs = plt.subplots(2, 1, figsize=(50, 30))
                if cfg.NCHANNELS==1: axs[0].imshow(im[:, :])
                else: axs[0].imshow(im[:, :,:])
                axs[0].set_xticks([])
                axs[0].set_yticks([])
                axs[0].text(0, 0, label)

                if cfg.NCHANNELS==1: pass#axs[1].imshow(im[:, :])
                else:axs[1].imshow(im[:, :, ::-1])
                axs[1].set_xticks([])
                axs[1].set_yticks([])

                plt.tight_layout()
                plt.show()
                plt.close()

            images.append(im)
            labels.append(label)

            if len(images) == batch_size:
                image_batch,label_vec,label_len,time_step = groupBatch(images,labels)
                yield image_batch,label_vec,label_len,time_step
                images = []
                labels = []
        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()
            continue

def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        enqueuer.start(max_queue_size=24, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()

if __name__ == '__main__':
    # gen = generator(batch_size=32, vis=False)
    gen = get_batch(num_workers=24,batch_size=32,vis=False)
    while True:
        images, labels,label_len,time_step =  next(gen)
        print(len(images)," ",images[0].shape)
