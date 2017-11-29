import tensorflow as tf
from .network import Network
from ..lstm.config import cfg


class LSTM_train(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.dtype = tf.float16 if cfg.USE_FLOAT16  else tf.float32
        self.mode = cfg.MODE

        self.data = tf.placeholder(tf.float32, shape=[None, None, cfg.NUM_FEATURES ], name='data') #N*t_s*features*channels
        self.labels = tf.placeholder(tf.int32,[None],name='labels')
        self.labels_align = tf.placeholder(tf.int32,[None,None],name='labels_align')
        self.time_step_len = tf.placeholder(tf.int32,[None], name='time_step_len')
        self.labels_len = tf.placeholder(tf.int32,[None],name='labels_len')

        self.keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
        self.layers = dict({'data': self.data,'labels':self.labels,
                            'time_step_len':self.time_step_len,
                            'labels_len':self.labels_len,
                            'keep_prob':self.keep_prob,
                            'labels_align':self.labels_align})
        self.trainable = trainable
        self.setup()

    def setup(self):
        (self.feed('data')
         .conv_single(3, 3, 64 ,1, 1, name='conv1',c_i=cfg.NCHANNELS)
         .conv_single(3, 3, 128 ,1, 1, name='conv2')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
         .conv_single(3, 3, 256 ,1, 1, name='conv3_1')
         .conv_single(3, 3, 256 ,1, 1, name='conv3_2')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
         .conv_single(3, 3, 256 ,1, 1, name='conv4_1', bn=True)
         .conv_single(3, 3, 256 ,1, 1, name='conv4_2', bn=True)
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
         .conv_single(1, 4, cfg.RNN.EMBEDD_SIZE ,1, 4, name='conv5', bn=True, relu=False)
         .reshape_squeeze_layer(d = cfg.RNN.EMBEDD_SIZE , name='reshaped_layer'))
        (self.feed('reshaped_layer','time_step_len', 'keep_prob')
         .encoder(rnn_cell_type = cfg.RNN.CELL_TYPE,hidden_units = cfg.RNN.HIDDEN_UNITS, use_dropout = cfg.RNN.USE_DROPOUT, use_residual = cfg.RNN.USE_RESIDUAL, name = 'encoder'))
        (self.feed('keep_prob','encoder','time_step_len','labels_align','labels_len')
         .attn_decoder(attention_type=cfg.RNN.ATTEN_TYPE, 
             attn_input_feeding=cfg.RNN.IS_ATTEN_INPUT_FEEDING, rnn_cell_type=cfg.RNN.CELL_TYPE,
             hidden_units=cfg.RNN.HIDDEN_UNITS,beam_width=cfg.RNN.BEAM_WIDTH, 
             use_beamsearch_decode = cfg.RNN.IS_USE_BEAMSEARCH, use_residual = cfg.RNN.USE_RESIDUAL,
             num_layers=cfg.RNN.DEPTH,num_decoder_symbols=cfg.NCLASSES, 
             use_dropout = cfg.RNN.USE_DROPOUT, embedding_size=cfg.RNN.EMBEDD_SIZE, name = 'loss_decode_res'))

         #.decoder(hidden_units = cfg.RNN.HIDDEN_UNITS, name = 'logits'))
         # .lstm(cfg.TRAIN.NUM_HID,cfg.TRAIN.NUM_LAYERS,name='logits',img_shape=[-1,cfg.IMG_SHAPE[0]//cfg.POOL_SCALE,cfg.NUM_FEATURES//cfg.POOL_SCALE]))
         #.bi_lstm(cfg.TRAIN.NUM_HID,cfg.TRAIN.NUM_LAYERS,name='logits',img_shape=[-1,cfg.IMG_SHAPE[0]//cfg.POOL_SCALE,cfg.NUM_FEATURES//cfg.POOL_SCALE]))
         #.bi_lstm(cfg.TRAIN.NUM_HID,cfg.TRAIN.NUM_LAYERS,name='logits'))
