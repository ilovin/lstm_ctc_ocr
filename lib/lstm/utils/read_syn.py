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
# from lib.utils.data_util import GeneratorEnqueuer
from keras.utils import GeneratorEnqueuer
from lib.lstm.config import cfg,get_encode_decode_dict
import tensorflow as tf
import cv2
import sys

def generateImg(img_path):
    img_name = img_path.split('/')[-1].split('_')[1]
    img = cv2.imread(img_path)
    theChars = img_name
    return img,theChars

encode_maps,decode_maps = get_encode_decode_dict()

def groupBatch(imgs,labels):
    max_w = -sys.maxsize
    time_steps = []
    label_align = []
    label_len = []
    label_vec = []
    img_batch = []
    height = cfg.IMG_HEIGHT

    #change all the image to the same height
    for i,img in enumerate(imgs):
        if cfg.NCHANNELS==1: h,w = img.shape
        else: h,w,_ = img.shape
        w = int(height/h*w)
        h = int(height)
        imgs[i] = cv2.resize(img,(w,h))

        max_w = max(max_w,w)
        time_step = w//cfg.POOL_SCALE
        if time_step <=0:
            print('time_step <=0:{}'.format(labels[i]))
            continue
        time_steps.append(w//cfg.POOL_SCALE)
        code = [encode_maps[c] for c in list(labels[i])]
        label_align.append(code)
        label_vec.extend(code)
        label_len.append(len(labels[i]))
    max_w = math.ceil(max_w/cfg.POOL_SCALE)*cfg.POOL_SCALE
    max_label_len = max(label_len)
    for i,ith_label in enumerate(label_align):
        label_align[i]+=[cfg.EOS_TOKEN]*(max_label_len-len(ith_label)) #padding with eos
    for img in imgs:
        if cfg.NCHANNELS==1: h,w = img.shape
        else: h,w,_ = img.shape
        img = cv2.copyMakeBorder(img,0,0,0,max_w-w,cv2.BORDER_CONSTANT,value=0).astype(np.float32)/255.
        img = img.swapaxes(0, 1)
        img = np.reshape(img,[-1,cfg.NCHANNELS*h])
        img_batch.append(img)
    #img_batch = np.array(img_batch)
    return img_batch,label_align,label_vec,label_len,time_steps

def generator(batch_size=32, vis=False, folder = cfg.SYN_FOLDER, txt_path = None):
    images = []
    labels = []
    img_paths = []
    annotation_txt = os.path.join(folder,txt_path)
    if not os.path.exists(annotation_txt):
        print(annotation_txt," does not exist")
    print(annotation_txt)
    with open(annotation_txt,'r') as f:
        for line in f:
            if not line:continue
            line = os.path.join(folder,line.split(' ')[0][2:])
            #print(line)
            img_paths.append(line)
    print("total train imgs: ", len(img_paths))

    while True:
        random.shuffle(img_paths)
        for img_path in img_paths:
            try:
                im, label = generateImg(img_path)
                h,w,_ = im.shape
                w = int(cfg.IMG_HEIGHT/h*w)
                time_step = w//cfg.POOL_SCALE
                if time_step <=0:
                    print('time_step <=0:{}'.format(img_path))
                    continue
                #img_size = cfg.IMG_SHAPE  # 160,60
                #im = cv2.resize(im,(img_size[0],img_size[1]))
                if cfg.NCHANNELS == 1:
                    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
                    #im = np.expand_dims(im,axis=2)
                # print(im.shape,' ',label)
                if vis:
                    print(label)
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
                    image_batch,label_align,label_vec,label_len,time_step = groupBatch(images,labels)
                    yield image_batch,label_align,label_vec,label_len,time_step
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
    gen = get_batch(num_workers=2,batch_size=4,folder = cfg.SYN_FOLDER,txt_path=cfg.TRAIN.TXT, vis=True)
    while True:
        images, labels,label_len,time_step =  next(gen)
        print(len(images)," ",images[0].shape)
