import sys,math
import os,shutil
import collections
import numpy as np
import os
import tensorflow as tf
import cv2
from lib.lstm.utils.timer import Timer
from ..lstm.config import cfg,get_encode_decode_dict

class SolverWrapper(object):
    def __init__(self, sess, network, imgdb, output_dir, logdir, pretrained_model=None):
        self.net = network
        self.imgdb = imgdb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model
        print('done')

        # For checkpoint
        self.saver = tf.train.Saver(max_to_keep=100)
        self.writer = tf.summary.FileWriter(logdir=logdir,
                                             graph=tf.get_default_graph(),
                                             flush_secs=5)



    def test_model(self,sess,testDir=None,restore = True):
        logits = self.net.get_output('logits')
        time_step_batch = self.net.get_output('time_step_len')
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, time_step_batch, merge_repeated=True)
        dense_decoded = tf.cast(tf.sparse_tensor_to_dense(decoded[0], default_value=0), tf.int32)

        img_size = cfg.IMG_SHAPE
        global_step = tf.Variable(0, trainable=False)
        # intialize variables
        local_vars_init_op = tf.local_variables_initializer()
        global_vars_init_op = tf.global_variables_initializer()

        combined_op = tf.group(local_vars_init_op, global_vars_init_op)
        sess.run(combined_op)
        # resuming a trainer
        if restore:
            try:
                ckpt = tf.train.get_checkpoint_state(self.output_dir)
                print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
                self.saver.restore(sess, tf.train.latest_checkpoint(self.output_dir))
                stem = os.path.splitext(os.path.basename(ckpt.model_checkpoint_path))[0]
                restore_iter = int(stem.split('_')[-1])
                sess.run(global_step.assign(restore_iter))
                print('done')
            except:
                raise Exception('Check your pretrained {:s}'.format(ckpt.model_checkpoint_path))

        timer = Timer()

        total = correct = 0
        for file in os.listdir(testDir):
            timer.tic()
            total+=1

            if cfg.NCHANNELS == 1: img = cv2.imread(os.path.join(testDir,file),0)
            else : img = cv2.imread(os.path.join(testDir,file),1)
            print(file,end=' ')
            #img = cv2.resize(img,tuple(img_size))
            w = img.shape[1]
            width = math.ceil(img.shape[1] / cfg.POOL_SCALE) * cfg.POOL_SCALE
            img = cv2.copyMakeBorder(img, 0, 0, 0, width - w, cv2.BORDER_CONSTANT, value=0).astype(np.float32) / 255.

            img = img.swapaxes(0,1)
            img = np.reshape(img, [1,width,cfg.NUM_FEATURES])
            #img = np.expand_dims(img,axis=0)
            feed_dict = {
                self.net.data: img,
                self.net.time_step_len: [img.shape[1]//cfg.POOL_SCALE],
                self.net.keep_prob: 1.0
            }
            res = sess.run(fetches=dense_decoded[0], feed_dict=feed_dict)
            def decodeRes(nums,ignore= 0):
                encode_maps,decode_maps = get_encode_decode_dict()
                res = [decode_maps[i] for i in nums if i!=ignore]
                return res
            org = file.split('.')[0].split('_')[1]
            res = ''.join(decodeRes(res))
            if org==res:correct+=1
            _diff_time = timer.toc(average=False)
            print('cost time: {:.3f},\n    res: {}'.format(_diff_time,res))
            #visualize_segmentation_adaptive(np.array(output),cls_dict)
        print('total acc:{}/{}={:.4f}'.format(correct,total,correct/total))


def test_net(network, imgdb, testDir, output_dir, log_dir, pretrained_model=None,restore=True):

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allocator_type = 'BFC'
    #config.gpu_options.per_process_gpu_memory_fraction = 0.4
    with tf.Session(config=config) as sess:
        sw = SolverWrapper(sess, network, imgdb, output_dir, logdir= log_dir, pretrained_model=pretrained_model)
        print('Solving...')
        sw.test_model(sess, testDir=testDir, restore=restore)
        print('done solving')

