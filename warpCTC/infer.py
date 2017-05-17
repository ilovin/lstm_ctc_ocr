import cv2,time,os,re
import tensorflow as tf
import numpy as np
import utils
import lstm_ocr as model_define

FLAGS = utils.FLAGS

inferFolder = 'tmp'
imgList = []
for root,subFolder,fileList in os.walk(inferFolder):
    for fName in fileList:
        if re.match(r'.*\.[jpg|png|jpeg]',fName.lower()):
            img_Path = os.path.join(root,fName)
            imgList.append(img_Path)


def main():
    g = model_define.Graph()
    with tf.Session(graph = g.graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=100)
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess,ckpt)
            print('restore from ckpt{}'.format(ckpt))
        else:
            print('cannot restore')
        
        imgStack = []
        right = total  = 0
        for img in imgList:
            pattern = r'.*_(.*)\..*'
            try:
                org = re.match(pattern,img).group(1)
            except:
                print('>>>>>>>>the img name does not match the pattern: ',img)
                continue
            total+=1
            im = cv2.imread(img,0).astype(np.float32)/255.
            im = cv2.resize(im,(utils.image_width,utils.image_height))
            im = im.swapaxes(0,1)
            imgStack.append(im)

            start = time.time()
            def get_input_lens(seqs):
                leghs = np.array([len(s) for s in seqs],dtype=np.int64)
                return seqs,leghs
            inp,seq_len = get_input_lens(np.array([im]))
            feed={g.inputs : inp,
                 g.seq_len : seq_len}
            d = sess.run(g.decoded[0],feed)
            dense_decoded = tf.sparse_tensor_to_dense(d,default_value=-1).eval(session=sess)
            res = '' 
            for d in dense_decoded:
                for i in d:
                    if i == -1:
                        res+=' '
                    else:
                        res+=utils.decode_maps[i]
            print('cost time: ',time.time()-start)
            if res == org.upper(): right+=1
            else: print('ORG: ',img,' decoded: ',res)
        print('total accuracy: ',right*1.0/total)


if __name__ == '__main__':
    main()
