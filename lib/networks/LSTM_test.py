import tensorflow as tf
from .network import Network
from ..lstm.config import cfg


class LSTM_test(Network):
    def __init__(self, trainable=True):
        self.inputs = []

        self.data = tf.placeholder(tf.float32, shape=[None, None, cfg.NUM_FEATURES], name='data')
        self.time_step_len = tf.placeholder(tf.int32,[None], name='time_step_len')

        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data': self.data, 'time_step_len':self.time_step_len})
        self.trainable = trainable
        self.setup()

    def setup(self):
        (self.feed('data','time_step_len')
         .lstm(cfg.TRAIN.NUM_HID,cfg.TRAIN.NUM_LAYERS,name='logits'))