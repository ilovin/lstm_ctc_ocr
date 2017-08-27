import numpy as np
import os,re
import tensorflow as tf
from ..lstm.config import cfg
from lib.lstm.utils.timer import Timer
from lib.lstm.utils.training import accuracy_calculation
from lib.lstm.utils.tf_records import read_tfrecord_and_decode_into_image_annotation_pair_tensors
from lib.lstm.utils.gen import get_batch

# charset = cfg.CHARSET
# SPACE_INDEX = 0
# SPACE_TOKEN = ''
# encode_maps = {}
# decode_maps = {}
# for i, char in enumerate(charset, 1):
#     encode_maps[char] = i
#     decode_maps[i] = char
# encode_maps[SPACE_TOKEN] = SPACE_INDEX
# decode_maps[SPACE_INDEX] = SPACE_TOKEN
class SolverWrapper(object):
    def __init__(self, sess, network, imgdb, pre_train,output_dir, logdir):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imgdb = imgdb
        self.pre_train=pre_train
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

    def train_model(self, sess, max_iters, restore=False):
        #img_b,lb_b,lb_len_b,t_s_b = self.get_data(self.imgdb.path,batch_size= cfg.TRAIN.BATCH_SIZE,num_epochs=cfg.TRAIN.NUM_EPOCHS)
        #val_img_b, val_lb_b, val_lb_len_b,val_t_s_b = self.get_data(self.imgdb.val_path,batch_size=cfg.VAL.BATCH_SIZE,num_epochs=cfg.VAL.NUM_EPOCHS)
        train_gen = get_batch(num_workers=12,batch_size=cfg.TRAIN.BATCH_SIZE,vis=False)
        val_gen = get_batch(num_workers=2,batch_size=cfg.VAL.BATCH_SIZE,vis=False)

        loss, dense_decoded = self.net.build_loss()

        tf.summary.scalar('loss', loss)
        summary_op = tf.summary.merge_all()

        # optimizer
        if cfg.TRAIN.SOLVER == 'Adam':
            opt = tf.train.AdamOptimizer(cfg.TRAIN.LEARNING_RATE)
            lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
        elif cfg.TRAIN.SOLVER == 'RMS':
            opt = tf.train.RMSPropOptimizer(cfg.TRAIN.LEARNING_RATE)
            lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
        else:
            lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
            momentum = cfg.TRAIN.MOMENTUM
            opt = tf.train.MomentumOptimizer(lr, momentum)

        global_step = tf.Variable(0, trainable=False)
        with_clip = True
        if with_clip:
            tvars = tf.trainable_variables()
            grads, norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), 10.0)
            train_op = opt.apply_gradients(list(zip(grads, tvars)), global_step=global_step)
        else:
            train_op = opt.minimize(loss, global_step=global_step)

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
        loss_min = 0.015
        first_val = True
        for iter in range(restore_iter, max_iters):
            timer.tic()
            # learning rate
            if iter != 0 and iter % cfg.TRAIN.STEPSIZE == 0:
                sess.run(tf.assign(lr, lr.eval() * cfg.TRAIN.GAMMA))

            # get one batch
            img_Batch,label_Batch, label_len_Batch,time_step_Batch = next(train_gen)
            img_Batch = np.array(img_Batch)
            # Subtract the mean pixel value from each pixel
            feed_dict = {
                self.net.data:          np.array(img_Batch),
                self.net.labels:        np.array(label_Batch),
                self.net.time_step_len: np.array(time_step_Batch),
                self.net.labels_len:    np.array(label_len_Batch),
                self.net.keep_prob:     0.5
            }

            fetch_list = [loss,summary_op,train_op]
            ctc_loss,summary_str, _ =  sess.run(fetches=fetch_list, feed_dict=feed_dict)

            self.writer.add_summary(summary=summary_str, global_step=global_step.eval())
            _diff_time = timer.toc(average=False)

            if (iter) % (cfg.TRAIN.DISPLAY) == 0:
                print('iter: %d / %d, total loss: %.7f, lr: %.7f'%\
                        (iter, max_iters, ctc_loss ,lr.eval()),end=' ')
                print('speed: {:.3f}s / iter'.format(_diff_time))
            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0 or ctc_loss<loss_min:
                if(ctc_loss<loss_min):
                    print('loss: ',ctc_loss,end=' ')
                    self.snapshot(sess, 1)
                    loss_min = ctc_loss
                else: self.snapshot(sess, iter)
            if (iter+1) % cfg.VAL.VAL_STEP == 0 or loss_min==ctc_loss:
                if first_val:
                    val_img_Batch,val_label_Batch, val_label_len_Batch,val_time_step_Batch = next(val_gen)
                    org = self.restoreLabel(val_label_Batch,val_label_len_Batch)
                    first_val=False

                feed_dict = {
                    self.net.data :          np.array(val_img_Batch),
                    self.net.labels :         np.array(val_label_Batch),
                    self.net.time_step_len : np.array(val_time_step_Batch),
                    self.net.labels_len :     np.array(val_label_len_Batch),
                    self.net.keep_prob:      1.0
                }

                # fetch_list = [dense_decoded]
                res =  sess.run(fetches=dense_decoded, feed_dict=feed_dict)
                acc = accuracy_calculation(org,res,ignore_value=0)
                print('accuracy: {:.5f}'.format(acc))


def train_net(network, imgdb, pre_train,output_dir, log_dir, max_iters=40000, restore=False):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allocator_type = 'BFC'
    with tf.Session(config=config) as sess:
        sw = SolverWrapper(sess, network, imgdb, pre_train,output_dir, logdir= log_dir)
        print('Solving...')
        sw.train_model(sess, max_iters, restore=restore)
        print('done solving')
