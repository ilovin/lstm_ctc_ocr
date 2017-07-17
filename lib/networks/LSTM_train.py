import tensorflow as tf
from .network import Network
from ..lstm.config import cfg


class LSTM_train(Network):
    def __init__(self, trainable=True):
        self.inputs = []

        self.data = tf.placeholder(tf.float32, shape=[None, None, cfg.NUM_FEATURES ], name='data') #N*t_s*features*channels
        self.labels = tf.placeholder(tf.int32,[None],name='labels')
        self.time_step_len = tf.placeholder(tf.int32,[None], name='time_step_len')
        self.labels_len = tf.placeholder(tf.int32,[None],name='labels_len')

        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data': self.data,'labels':self.labels,
                            'time_step_len':self.time_step_len,
                            'labels_len':self.labels_len})
        self.trainable = trainable
        self.setup()

    def setup(self):
        (self.feed('data')
         .conv_single(5, 5, 32 ,1, 1, name='conv1',c_i=cfg.NCHANNELS)
         .conv_single(3, 3, 32 ,1, 1, name='conv2')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
         .conv_single(3, 3, 1 ,1, 1, name='conv3',relu=False))
        (self.feed('conv3','time_step_len')
         .lstm(cfg.TRAIN.NUM_HID,cfg.TRAIN.NUM_LAYERS,name='logits',img_shape=[-1,cfg.IMG_SHAPE[0]//2,cfg.NUM_FEATURES//2]))
         # .bi_lstm(cfg.TRAIN.NUM_HID,cfg.TRAIN.NUM_LAYERS,name='logits'))
