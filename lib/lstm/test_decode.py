import numpy as np
import os,re
import tensorflow as tf
from ..lstm.config import cfg,get_encode_decode_dict
from lib.lstm.utils.timer import Timer
from lib.lstm.utils.training import accuracy_calculation,attn_accuracy_calculation
from lib.lstm.utils.tf_records import read_tfrecord_and_decode_into_image_annotation_pair_tensors
from lib.lstm.utils.read_gen import get_batch,generator

encode_maps,decode_maps = get_encode_decode_dict()
class SolverWrapper(object):
    def __init__(self, sess, network, imgdb, output_dir, logdir):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imgdb = imgdb
        self.output_dir = output_dir
        print('done')
        self.saver = tf.train.Saver(max_to_keep=100)
        self.writer = tf.summary.FileWriter(logdir=logdir,
                                             graph=tf.get_default_graph(),
                                             flush_secs=5)

    def snapshot(self, sess, iter):
        net = self.net
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + '_ctc' + infix +
                        '_iter_{:d}'.format(iter + 1) + '.ckpt')
        
        #filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
         #           '_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)
        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

    def get_data(self,path,batch_size,num_epochs):
        filename_queue = tf.train.string_input_producer([path], num_epochs=num_epochs)
        image,label,label_len,time_step= read_tfrecord_and_decode_into_image_annotation_pair_tensors(filename_queue)
        image_batch, label_batch, label_len_batch,time_step_batch = tf.train.shuffle_batch([image,label,label_len,time_step],
                                                                                           batch_size=batch_size,
                                                                                           capacity=9600,
                                                                                           num_threads=4,
                                                                                           min_after_dequeue=6400)
        return image_batch, label_batch, label_len_batch,time_step_batch

    def restoreLabel(self,label_vec,label_len):
        labels = []
        for l_len in label_len:
            labels.append(label_vec[:l_len])
            label_vec = label_vec[l_len:]
        return labels

    def mergeLabel(self,labels,ignore = 0):
        label_lst = []
        for l in labels:
            while l[-1] == ignore: l = l[:-1]
            label_lst.extend(l)
        return np.array(label_lst)

    def validation(self, sess,dense_decoded, val_img_Batch,val_label_align_Batch, val_label_Batch, 
            val_label_len_Batch,val_time_step_Batch):
        org = self.restoreLabel(val_label_Batch,val_label_len_Batch)
        feed_dict = {
            self.net.data :           np.array(val_img_Batch),
            self.net.time_step_len :  np.array(val_time_step_Batch),
            self.net.labels_align:    np.array(val_label_align_Batch),
            self.net.labels :         np.array(val_label_Batch),
            self.net.labels_len :     np.array(val_label_len_Batch),
            self.net.keep_prob:       1.0
        }

        # fetch_list = [dense_decoded]
        res = sess.run(fetches=dense_decoded, feed_dict=feed_dict)
        acc = attn_accuracy_calculation(org,res,eos_token=encode_maps['>'])
        print('accuracy: {:.5f}'.format(acc))



    def test_model(self, sess, max_iters, restore=False):
        #img_b,lb_b,lb_len_b,t_s_b = self.get_data(self.imgdb.path,batch_size= cfg.TRAIN.BATCH_SIZE,num_epochs=cfg.TRAIN.NUM_EPOCHS)
        #val_img_b, val_lb_b, val_lb_len_b,val_t_s_b = self.get_data(self.imgdb.val_path,batch_size=cfg.VAL.BATCH_SIZE,num_epochs=cfg.VAL.NUM_EPOCHS)
        #multi thread
        val_gen = get_batch(num_workers=2,batch_size=cfg.VAL.BATCH_SIZE, vis=False,
                folder = cfg.SYN_FOLDER, txt_path = cfg.VAL.TXT)

        loss, dense_decoded = self.net.build_loss()

        global_step = tf.Variable(0, trainable=False)

        # intialize variables
        local_vars_init_op = tf.local_variables_initializer()
        global_vars_init_op = tf.global_variables_initializer()

        combined_op = tf.group(local_vars_init_op, global_vars_init_op)
        sess.run(combined_op)
        restore_iter = 1

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
        first_val = True
        restore_iter = 1
        #if self.mode == 'decode':
        for iter in range(restore_iter, max_iters):
            timer.tic()
            # learning rate
            # get one batch
            img_Batch,label_align_Batch, label_Batch, label_len_Batch,time_step_Batch = next(val_gen)
            self.validation(sess, dense_decoded, img_Batch, label_align_Batch,
                    label_Batch, label_len_Batch,time_step_Batch)
            _diff_time = timer.toc(average=False)
            print('cost time: {}'.format(_diff_time))

def test_net(network, imgdb, output_dir, log_dir, max_iters=40000, restore=False):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = cfg.GPU_USAGE
    config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type = 'BFC'
    with tf.Session(config=config) as sess:
        sw = SolverWrapper(sess, network, imgdb, output_dir, logdir= log_dir)
        print('Solving...')
        sw.test_model(sess, max_iters, restore=restore)
        print('done solving')
