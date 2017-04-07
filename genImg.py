import sys, random,os
import numpy as np
from captcha.image import ImageCaptcha
import cv2

def randRGB():
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

#10+26+26
char_set='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
def gen_rand():
    buf = ""
    max_len = random.randint(4,6)
    for i in range(max_len):
       buf += random.choice(char_set)
    return buf
def run(num,path):
    captcha=ImageCaptcha(fonts=['./fonts/Ubuntu-M.ttf'])
    if not os.path.exists(path):
        os.mkdir(path)
    for i in range(num):
        theChars=gen_rand()
        data=captcha.generate(theChars)
        img_name= '{:08d}'.format(i)+'_'+theChars+'.png'
        img_path=path+'/'+img_name
        captcha.write(theChars,img_path)
        print(img_path)

if __name__=='__main__':
    run(64*2000,'train')
    run(400,'test')
