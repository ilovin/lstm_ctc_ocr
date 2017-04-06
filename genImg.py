import sys, random,os
import numpy as np
from captcha.image import ImageCaptcha
import cv2

def randRGB():
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

char_set='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
def gen_rand():
    buf = ""
    max_len = random.randint(4,8)
    for i in range(max_len):
       buf += char_set[random.randint(0,61)]
    return buf
def run(num,path):
    captcha=ImageCaptcha(fonts=['./fonts/Ubuntu-M.ttf'])
    if not os.path.exists(path):
        os.mkdir(path)
    for i in range(num):
        theChars=gen_rand()
        data=captcha.generate(theChars)
        img_name= '{:05d}'.format(i)+'_'+theChars+'.png'
        img_path=path+'/'+img_name
        captcha.write(theChars,img_path)
        print(img_path)

if __name__=='__main__':
    run(64*200,'train')
    #run(500,'test_char')
