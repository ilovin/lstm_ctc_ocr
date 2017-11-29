import math
import numpy as np
import tensorflow as tf

import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.ops.rnn_cell import GRUCell, LSTMCell, MultiRNNCell
from tensorflow.python.ops.rnn_cell import DropoutWrapper, ResidualWrapper

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest

from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder

from lib.lstm.config import cfg,get_encode_decode_dict
from lib.lstm.utils.training import *
import warpctc_tensorflow

DEFAULT_PADDING = 'SAME'

def incluude_original(dec):
    """ Meta decorator, which make the original function callable (via f._original() )"""
    def meta_decorator(f):
        decorated = dec(f)
        decorated._original = f
        return decorated
    return meta_decorator

#@include_original
def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs)==0:
            raise RuntimeError('No input variables found for layer %s.'%name)
        elif len(self.inputs)==1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self
    return layer_decorated

class Network(object):
    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, session, ignore_missing=False):
        data_dict = np.load(data_path,encoding='latin1').item()
        for key in data_dict:
            with tf.variable_scope(key, reuse=True):
                for subkey in data_dict[key]:
                    try:
                        var = tf.get_variable(subkey)
                        session.run(var.assign(data_dict[key][subkey]))
                        print("assign pretrain model "+subkey+ " to "+key)
                    except ValueError:
                        print("ignore "+key)
                        if not ignore_missing:

                            raise

    def feed(self, *args):
        assert len(args)!=0
        self.inputs = []
        for layer in args:
            if isinstance(layer, str):
                try:
                    layer = self.layers[layer]
                    print(layer)
                except KeyError:
                    print(list(self.layers.keys()))
                    raise KeyError('Unknown layer name fed: %s'%layer)
            self.inputs.append(layer)
        return self

    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print(list(self.layers.keys()))
            raise KeyError('Unknown layer name fed: %s'%layer)
        return layer

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t,_ in list(self.layers.items()))+1
        return '%s_%d'%(prefix, id)

    def make_var(self, name, shape, initializer=None, trainable=True, regularizer=None):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, regularizer=regularizer)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    @layer
    def bi_lstm(self, input, num_hids, num_layers, name,img_shape = None ,trainable=True):
        img,img_len = input[0],input[1]
        padding = tf.constant([[0,0],[1,0],[0,0]])#only padding left to zero,padding the time_step dimension
        #img = tf.squeeze(img,axis=3)
        if img_shape:img =tf.reshape(img,shape = img_shape )
        img = tf.pad(img,padding,"CONSTANT")
        img_len+=1
        with tf.variable_scope(name) as scope:
            #stack = tf.contrib.rnn.MultiRNNCell([cell,cell1] , state_is_tuple=True)
            lstm_fw_cell = tf.contrib.rnn.LSTMCell(num_hids//2,state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.LSTMCell(num_hids//2,state_is_tuple=True)

            output,_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,img,img_len,dtype=tf.float32)
            # output_bw_reverse = tf.reverse_sequence(output[1],img_len,seq_axis=1)
            output = tf.concat(output,axis=2)

            stack_cell = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.LSTMCell(num_hids, state_is_tuple=True) for _ in range(num_layers)],
                state_is_tuple=True)
            stack_cell = tf.contrib.rnn.AttentionCellWrapper(stack_cell, attn_length = 5)
            lstm_out,last_state = tf.nn.dynamic_rnn(stack_cell,output,img_len,dtype=tf.float32)
            shape = tf.shape(img)
            batch_size, time_step = shape[0],shape[1]
            lstm_out = tf.reshape(lstm_out,[-1,num_hids])
            init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
            # init_weights = tf.contrib.layers.xavier_initializer()
            # init_weights = tf.truncated_normal_initializer(stddev=0.1)
            init_biases = tf.constant_initializer(0.0)
            W = self.make_var('weights', [num_hids, cfg.NCLASSES], init_weights, trainable, \
                              regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            b = self.make_var('biases', [cfg.NCLASSES], init_biases, trainable)
            logits = tf.matmul(lstm_out,W)+b
            logits = tf.reshape(logits,[batch_size,-1,cfg.NCLASSES])
            logits = tf.transpose(logits,(1,0,2))
            return logits
    @layer
    def lstm(self, input, num_hids, num_layers, name,img_shape = None ,trainable=True):
        img,img_len = input[0],input[1]
        if img_shape:img =tf.reshape(img,shape = img_shape )
        with tf.variable_scope(name) as scope:
            encoder_cell = tf.contrib.rnn.LSTMCell(num_hids,state_is_tuple=True)
            encoder_output,last_state = tf.nn.dynamic_rnn(encoder_cell,img,img_len,dtype=tf.float32)

            stack_cell = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.LSTMCell(num_hids, state_is_tuple=True) for _ in range(num_layers)],
                state_is_tuple=True)
            lstm_out,last_state = tf.nn.dynamic_rnn(stack_cell,encoder_output,img_len,dtype=tf.float32)
            shape = tf.shape(img)
            batch_size, time_step = shape[0],shape[1]
            lstm_out = tf.reshape(lstm_out,[-1,num_hids])
            # init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.001, mode='FAN_AVG', uniform=False)
            # init_weights = tf.contrib.layers.xavier_initializer()
            init_weights = tf.truncated_normal_initializer(stddev=0.1)
            init_biases = tf.constant_initializer(0.0)
            W = self.make_var('weights', [num_hids, cfg.NCLASSES], init_weights, trainable, \
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            b = self.make_var('biases', [cfg.NCLASSES], init_biases, trainable)
            logits = tf.matmul(lstm_out,W)+b
            logits = tf.reshape(logits,[batch_size,-1,cfg.NCLASSES])
            logits = tf.transpose(logits,(1,0,2))
            return logits

    @layer
    def concat(self, input, axis, name):
        with tf.variable_scope(name) as scope:
            concat = tf.concat(values=input,axis=axis)
        return concat

    @layer
    def conv_single(self, input, k_h, k_w, c_o, s_h, s_w, name, c_i=None, bn=False, biased=True,relu=True, padding=DEFAULT_PADDING, trainable=True):
        """ contribution by miraclebiu, and biased option"""
        self.validate_padding(padding)
        if not c_i: c_i = input.get_shape()[-1]
        if c_i==1: input = tf.expand_dims(input=input,axis=3)
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1,s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            init_weights = tf.contrib.layers.xavier_initializer()
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i, c_o], init_weights, trainable, \
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            if biased:
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                conv = convolve(input, kernel)
                bias = tf.nn.bias_add(conv, biases)
                if bn:
                    bn_layer = tf.contrib.layers.batch_norm(bias, scale=True,
                                                            center=True, is_training=True, scope=name)
                else:bn_layer = bias
                if relu:
                    return tf.nn.relu(bn_layer)
                else: return bn_layer
            else:
                conv = convolve(input, kernel)
                if bn:
                    bn_layer = tf.contrib.layers.batch_norm(conv, scale=True,
                                                            center=True, is_training=True, scope=name)
                else:bn_layer = conv
                if relu:
                    return tf.nn.relu(bn_layer)
                return bn_layer

    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, c_i=None, biased=True,relu=True, padding=DEFAULT_PADDING, trainable=True):
        """ contribution by miraclebiu, and biased option"""
        self.validate_padding(padding)
        if not c_i: c_i = input.get_shape()[-1]
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            init_weights = tf.contrib.layers.xavier_initializer()
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i, c_o], init_weights, trainable, \
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            if biased:
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                conv = convolve(input, kernel)
                if relu:
                    bias = tf.nn.bias_add(conv, biases)

                    return tf.nn.relu(bias)
                return tf.nn.bias_add(conv, biases)
            else:
                conv = convolve(input, kernel)
                if relu:
                    return tf.nn.relu(conv)
                return conv

    @layer
    def conv_zero(self, input, k_h, k_w, c_o, s_h, s_w, name, biased=True, relu=True, padding=DEFAULT_PADDING,
             trainable=True):
        """ contribution by miraclebiu, and biased option"""
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            init_weights = tf.constant_initializer(0.0)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i, c_o], init_weights, trainable, \
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            if biased:
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                conv = convolve(input, kernel)
                if relu:
                    bias = tf.nn.bias_add(conv, biases)

                    return tf.nn.relu(bias)
                return tf.nn.bias_add(conv, biases)
            else:
                conv = convolve(input, kernel)
                if relu:
                    return tf.nn.relu(conv)
                return conv

    @layer
    def conv_norm(self, input, k_h, k_w, c_o, s_h, s_w, name, biased=True,relu=True, padding=DEFAULT_PADDING, trainable=True):
        """ contribution by miraclebiu, and biased option"""
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.001, mode='FAN_AVG', uniform=False)
            # init_weights = tf.contrib.layers.xavier_initializer()
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i, c_o], init_weights, trainable, \
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            if biased:
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                conv = convolve(input, kernel)
                if relu:
                    bias = tf.nn.bias_add(conv, biases)
                    temp_layer = tf.contrib.layers.batch_norm(bias, scale=True, center=True, is_training=True,
                                                      scope=name)
                    return tf.nn.relu(temp_layer)
                return tf.nn.bias_add(conv, biases)
            else:
                conv = convolve(input, kernel)
                if relu:
                    return tf.nn.crelu(conv)
                return conv

    @layer
    def conv_final(self, input, k_h, k_w, c_o, s_h, s_w, name, biased=True, relu=True, padding=DEFAULT_PADDING,
                  trainable=True):
        """ contribution by miraclebiu, and biased option"""
        self.validate_padding(padding)
        c_i = 128
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.001, mode='FAN_AVG', uniform=False)
            # init_weights = tf.contrib.layers.xavier_initializer()
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i, c_o], init_weights, trainable, \
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            if biased:
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                conv = convolve(input, kernel)
                if relu:
                    bias = tf.nn.bias_add(conv, biases)
                    temp_layer = tf.contrib.layers.batch_norm(bias, scale=True, center=True, is_training=True,
                                                              scope=name)
                    return tf.nn.relu(temp_layer)
                return tf.nn.bias_add(conv, biases)
            else:
                conv = convolve(input, kernel)
                if relu:
                    return tf.nn.crelu(conv)
                return conv

    @layer
    def upconv(self, input, shape, c_o, ksize=4, stride = 2, name = 'upconv', biased=False, relu=True, padding=DEFAULT_PADDING,
             trainable=True):
        """ up-conv"""
        self.validate_padding(padding)

        c_in = input.get_shape()[3].value
        in_shape = tf.shape(input)
        if shape is None:
            h = ((in_shape[1] ) * stride)
            w = ((in_shape[2] ) * stride)
            new_shape = [in_shape[0], h, w, c_o]
        else:
            new_shape = [in_shape[0], shape[1], shape[2], c_o]
        output_shape = tf.stack(new_shape)

        filter_shape = [ksize, ksize, c_o, c_in]

        with tf.variable_scope(name) as scope:
            # init_weights = tf.contrib.layers.xavier_initializer()
            init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.001, mode='FAN_AVG', uniform=False)
            filters = self.make_var('weights', filter_shape, init_weights, trainable, \
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            deconv = tf.nn.conv2d_transpose(input, filters, output_shape,
                                            strides=[1, stride, stride, 1], padding=DEFAULT_PADDING, name=scope.name)
            # coz de-conv losses shape info, use reshape to re-gain shape
            deconv = tf.reshape(deconv, new_shape)

            if biased:
                init_biases = tf.constant_initializer(0.0)
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                if relu:
                    bias = tf.nn.bias_add(deconv, biases)
                    return tf.nn.relu(bias)
                return tf.nn.bias_add(deconv, biases)
            else:
                if relu:
                    return tf.nn.relu(deconv)
                return deconv

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def reshape_layer(self, input, d, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob_reshape':
            #
            # transpose: (1, AxH, W, 2) -> (1, 2, AxH, W)
            # reshape: (1, 2xA, H, W)
            # transpose: -> (1, H, W, 2xA)
             return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),
                                            [   input_shape[0],
                                                int(d),
                                                tf.cast(tf.cast(input_shape[1],tf.float32)/tf.cast(d,tf.float32)*tf.cast(input_shape[3],tf.float32),tf.int32),
                                                input_shape[2]
                                            ]),
                                 [0,2,3,1],name=name)
        else:
             return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),
                                        [   input_shape[0],
                                            int(d),
                                            tf.cast(tf.cast(input_shape[1],tf.float32)*(tf.cast(input_shape[3],tf.float32)/tf.cast(d,tf.float32)),tf.int32),
                                            input_shape[2]
                                        ]),
                                 [0,2,3,1],name=name)

    @layer
    def reshape_squeeze_layer(self, input, d, name):
        #N,H,W,C-> N,H*W,C
        input_shape = tf.shape(input)
        return tf.reshape(input, \
                          [input_shape[0], \
                           input_shape[1]*input_shape[2], \
                           int(d)])

    @layer
    def spatial_reshape_layer(self, input, d, name):
        input_shape = tf.shape(input)
        # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
        return tf.reshape(input,\
                               [input_shape[0],\
                                input_shape[1], \
                                -1,\
                                int(d)])


    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)


    @layer
    def fc(self, input, num_out, name, relu=True, trainable=True):
        with tf.variable_scope(name) as scope:
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]

            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(tf.transpose(input,[0,3,1,2]), [-1, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))

            if name == 'bbox_pred':
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
                init_biases = tf.constant_initializer(0.0)
            else:
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
                init_biases = tf.constant_initializer(0.0)

            weights = self.make_var('weights', [dim, num_out], init_weights, trainable, \
                                    regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            biases = self.make_var('biases', [num_out], init_biases, trainable)

            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob':
            return tf.reshape(tf.nn.softmax(tf.reshape(input,[-1,input_shape[3]])),[-1,input_shape[1],input_shape[2],input_shape[3]],name=name)
        else:
            return tf.nn.softmax(input,name=name)

    @layer
    def spatial_softmax(self, input, name):
        input_shape = tf.shape(input)
        # d = input.get_shape()[-1]
        return tf.reshape(tf.nn.softmax(tf.reshape(input, [-1, input_shape[3]])),
                          [-1, input_shape[1], input_shape[2], input_shape[3]], name=name)

    @layer
    def add(self,input,name):
        """contribution by miraclebiu"""
        return tf.add(input[0],input[1], name=name)

    @layer
    def batch_normalization(self,input,name,relu=True, is_training=False):
        """contribution by miraclebiu"""
        if relu:
            temp_layer=tf.contrib.layers.batch_norm(input,scale=True,center=True,is_training=is_training,scope=name)
            return tf.nn.relu(temp_layer)
        else:
            return tf.contrib.layers.batch_norm(input,scale=True,center=True,is_training=is_training,scope=name)

    @layer
    def negation(self, input, name):
        """ simply multiplies -1 to the tensor"""
        return tf.multiply(input, -1.0, name=name)

    @layer
    def bn_scale_combo(self, input, c_in, name, relu=True):
        """ PVA net BN -> Scale -> Relu"""
        with tf.variable_scope(name) as scope:
            bn = self.batch_normalization._original(self, input, name='bn', relu=False, is_training=False)
            # alpha = tf.get_variable('bn_scale/alpha', shape=[c_in, ], dtype=tf.float32,
            #                     initializer=tf.constant_initializer(1.0), trainable=True,
            #                     regularizer=self.l2_regularizer(0.00001))
            # beta = tf.get_variable('bn_scale/beta', shape=[c_in, ], dtype=tf.float32,
            #                    initializer=tf.constant_initializer(0.0), trainable=True,
            #                    regularizer=self.l2_regularizer(0.00001))
            # bn = tf.add(tf.mul(bn, alpha), beta)
            if relu:
                bn = tf.nn.relu(bn, name='relu')
            return bn

    @layer
    def pva_negation_block(self, input, k_h, k_w, c_o, s_h, s_w, name, biased=True, padding=DEFAULT_PADDING, trainable=True,
                           scale = True, negation = True):
        """ for PVA net, Conv -> BN -> Neg -> Concat -> Scale -> Relu"""
        with tf.variable_scope(name) as scope:
            conv = self.conv._original(self, input, k_h, k_w, c_o, s_h, s_w, biased=biased, relu=False, name='conv', padding=padding, trainable=trainable)
            conv = self.batch_normalization._original(self, conv, name='bn', relu=False, is_training=False)
            c_in = c_o
            if negation:
                conv_neg = self.negation._original(self, conv, name='neg')
                conv = tf.concat(axis=3, values=[conv, conv_neg], name='concat')
                c_in += c_in
            if scale:
                # y = \alpha * x + \beta
                alpha = tf.get_variable('scale/alpha', shape=[c_in,], dtype=tf.float32,
                                        initializer=tf.constant_initializer(1.0), trainable=True, regularizer=self.l2_regularizer(0.00001))
                beta = tf.get_variable('scale/beta', shape=[c_in, ], dtype=tf.float32,
                                        initializer=tf.constant_initializer(0.0), trainable=True, regularizer=self.l2_regularizer(0.00001))
                # conv = conv * alpha + beta
                conv = tf.add(tf.multiply(conv, alpha), beta)
            return tf.nn.relu(conv, name='relu')

    @layer
    def pva_negation_block_v2(self, input, k_h, k_w, c_o, s_h, s_w, c_in, name, biased=True, padding=DEFAULT_PADDING, trainable=True,
                           scale = True, negation = True):
        """ for PVA net, BN -> [Neg -> Concat ->] Scale -> Relu -> Conv"""
        with tf.variable_scope(name) as scope:
            bn = self.batch_normalization._original(self, input, name='bn', relu=False, is_training=False)
            if negation:
                bn_neg = self.negation._original(self, bn, name='neg')
                bn = tf.concat(axis=3, values=[bn, bn_neg], name='concat')
                c_in += c_in
                # y = \alpha * x + \beta
                alpha = tf.get_variable('scale/alpha', shape=[c_in,], dtype=tf.float32,
                                        initializer=tf.constant_initializer(1.0), trainable=True, regularizer=self.l2_regularizer(0.00004))
                beta = tf.get_variable('scale/beta', shape=[c_in, ], dtype=tf.float32,
                                        initializer=tf.constant_initializer(0.0), trainable=True, regularizer=self.l2_regularizer(0.00004))
                bn = tf.add(tf.multiply(bn, alpha), beta)
            bn = tf.nn.relu(bn, name='relu')
            if name == 'conv3_1/1': self.layers['conv3_1/1/relu'] = bn

            conv = self.conv._original(self, bn, k_h, k_w, c_o, s_h, s_w, biased=biased, relu=False, name='conv', padding=padding,
                         trainable=trainable)
            return conv

    @layer
    def pva_inception_res_stack(self, input, c_in, name, block_start = False, type = 'a'):

        if type == 'a':
            (c_0, c_1, c_2, c_pool, c_out) = (64, 64, 24, 128, 256)
        elif type == 'b':
            (c_0, c_1, c_2, c_pool, c_out) = (64, 96, 32, 128, 384)
        else:
            raise ('Unexpected inception-res type')
        if block_start:
            stride = 2
        else:
            stride = 1
        with tf.variable_scope(name+'/incep') as scope:
            bn = self.batch_normalization._original(self, input, name='bn', relu=False, is_training=False)
            bn_scale = self.scale._original(self, bn, c_in, name='bn_scale')
            ## 1 x 1

            conv = self.conv._original(self, bn_scale, 1, 1, c_0, stride, stride, name='0/conv', biased = False, relu=False)
            conv_0 = self.bn_scale_combo._original(self, conv, c_in=c_0, name ='0', relu=True)

            ## 3 x 3
            bn_relu = tf.nn.relu(bn_scale, name='relu')
            if name == 'conv4_1': tmp_c = c_1; c_1 = 48
            conv = self.conv._original(self, bn_relu, 1, 1, c_1, stride, stride, name='1_reduce/conv', biased = False, relu=False)
            conv = self.bn_scale_combo._original(self, conv, c_in=c_1, name='1_reduce', relu=True)
            if name == 'conv4_1': c_1 = tmp_c
            conv = self.conv._original(self, conv, 3, 3, c_1 * 2, 1, 1, name='1_0/conv', biased = False, relu=False)
            conv_1 = self.bn_scale_combo._original(self, conv, c_in=c_1 * 2, name='1_0', relu=True)

            ## 5 x 5
            conv = self.conv._original(self, bn_scale, 1, 1, c_2, stride, stride, name='2_reduce/conv', biased = False, relu=False)
            conv = self.bn_scale_combo._original(self, conv, c_in=c_2, name='2_reduce', relu=True)
            conv = self.conv._original(self, conv, 3, 3, c_2 * 2, 1, 1, name='2_0/conv', biased = False, relu=False)
            conv = self.bn_scale_combo._original(self, conv, c_in=c_2 * 2, name='2_0', relu=True)
            conv = self.conv._original(self, conv, 3, 3, c_2 * 2, 1, 1, name='2_1/conv', biased = False, relu=False)
            conv_2 = self.bn_scale_combo._original(self, conv, c_in=c_2 * 2, name='2_1', relu=True)

            ## pool
            if block_start:
                pool = self.max_pool._original(self, bn_scale, 3, 3, 2, 2, padding=DEFAULT_PADDING, name='pool')
                pool = self.conv._original(self, pool, 1, 1, c_pool, 1, 1, name='poolproj/conv', biased = False, relu=False)
                pool = self.bn_scale_combo._original(self, pool, c_in=c_pool, name='poolproj', relu=True)

        with tf.variable_scope(name) as scope:
            if block_start:
                concat = tf.concat(axis=3, values=[conv_0, conv_1, conv_2, pool], name='concat')
                proj = self.conv._original(self, input, 1, 1, c_out, 2, 2, name='proj', biased=True,
                                           relu=False)
            else:
                concat = tf.concat(axis=3, values=[conv_0, conv_1, conv_2], name='concat')
                proj = input

            conv = self.conv._original(self, concat, 1, 1, c_out, 1, 1, name='out/conv', relu=False)
            if name == 'conv5_4':
                conv = self.bn_scale_combo._original(self, conv, c_in=c_out, name='out', relu=False)
            conv = self.add._original(self, [conv, proj], name='sum')
        return  conv

    @layer
    def pva_inception_res_block(self, input, name, name_prefix = 'conv4_', type = 'a'):
        """build inception block"""
        node = input
        if type == 'a':
            c_ins = (128, 256, 256, 256, 256, )
        else:
            c_ins = (256, 384, 384, 384, 384, )
        for i in range(1, 5):
            node = self.pva_inception_res_stack._original(self, node, c_in = c_ins[i-1],
                                                          name = name_prefix + str(i), block_start=(i==1), type=type)
        return node

    @layer
    def scale(self, input, c_in, name):
        with tf.variable_scope(name) as scope:

            alpha = tf.get_variable('alpha', shape=[c_in, ], dtype=tf.float32,
                                    initializer=tf.constant_initializer(1.0), trainable=True,
                                    regularizer=self.l2_regularizer(0.00001))
            beta = tf.get_variable('beta', shape=[c_in, ], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0), trainable=True,
                                   regularizer=self.l2_regularizer(0.00001))
            return tf.add(tf.multiply(input, alpha), beta)


    @layer
    def dropout(self, input, keep_prob, name):
        return tf.nn.dropout(input, keep_prob, name=name)

    def l2_regularizer(self, weight_decay=0.0005, scope=None):
        def regularizer(tensor):
            with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
                l2_weight = tf.convert_to_tensor(weight_decay,
                                       dtype=tensor.dtype.base_dtype,
                                       name='weight_decay')
                return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
        return regularizer

    def smooth_l1_dist(self, deltas, sigma2=9.0, name='smooth_l1_dist'):
        with tf.name_scope(name=name) as scope:
            deltas_abs = tf.abs(deltas)
            smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0/sigma2), tf.float32)
            return tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign + \
                        (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)

    def build_single_cell(self,keep_prob_placehoder, rnn_cell_type,hidden_units, use_dropout, use_residual):
        assert cfg.RNN.CELL_TYPE in ['lstm','gru','LSTM','GRU']
        if (cfg.RNN.CELL_TYPE.lower() == 'gru'): cell_type = GRUCell
        else: cell_type = LSTMCell

        cell = cell_type(hidden_units)

        if use_dropout:
            cell = DropoutWrapper(cell, dtype=self.dtype,
                                  output_keep_prob=keep_prob_placehoder,)
        if use_residual:
            cell = ResidualWrapper(cell)
            
        return cell

    # Building encoder cell
    def build_encoder_cell (self,keep_prob_placehoder, rnn_cell_type,hidden_units,use_dropout = True, use_residual = False, num_layers = 2):
        return MultiRNNCell([self.build_single_cell(keep_prob_placehoder,rnn_cell_type = rnn_cell_type,hidden_units = hidden_units,use_dropout = use_dropout, use_residual = use_residual) for i in range(num_layers)])

    @layer
    def encoder(self,input, name,rnn_cell_type,hidden_units,use_dropout = True, use_residual = False, num_layers=2):
        print("building encoder..")
        encoder_inputs_embedded,encoder_inputs_length,keep_prob_placehoder = input
        with tf.variable_scope(name) as scope:
            # Building encoder_cell
            encoder_cell = self.build_encoder_cell(keep_prob_placehoder,rnn_cell_type = rnn_cell_type,hidden_units = hidden_units,use_dropout = use_dropout, use_residual = use_residual, num_layers = num_layers)
            #original encoder for word -> embedding
            ## Initialize encoder_embeddings to have variance=1.
            #sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            #initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=self.dtype)
            # 
            #self.encoder_embeddings = tf.get_variable(name='embedding',
            #    shape=[self.num_encoder_symbols, self.embedding_size],
            #    initializer=initializer, dtype=self.dtype)
            #
            ## Embedded_inputs: [batch_size, time_step, embedding_size]
            #self.encoder_inputs_embedded = tf.nn.embedding_lookup(
            #    params=self.encoder_embeddings, ids=self.encoder_inputs)
       
            # Input projection layer to feed embedded inputs to the cell
            # ** Essential when use_residual=True to match input/output dims
            input_layer = Dense(hidden_units, dtype=self.dtype, name='input_projection')
            # Embedded inputs having gone through input projection layer
            encoder_inputs_embedded = input_layer(encoder_inputs_embedded)
    
            # Encode input sequences into context vectors:
            # encoder_outputs: [batch_size, max_time_step, cell_output_size]
            # encoder_state: [batch_size, cell_output_size]
            encoder_outputs, encoder_last_state = tf.nn.dynamic_rnn(
                cell=encoder_cell, inputs=encoder_inputs_embedded,
                sequence_length=encoder_inputs_length, dtype=self.dtype,
                time_major=False)
            return encoder_outputs, encoder_last_state

    # Building decoder cell and attention. Also returns decoder_initial_state
    def build_decoder_cell(self, keep_prob_placehoder,
            encoder_outputs,encoder_last_state,encoder_inputs_length,
            attention_type,attn_input_feeding,rnn_cell_type,hidden_units,beam_width,
            use_dropout = True, use_beamsearch_decode = True, use_residual = False, num_layers=2):

        #encoder_outputs = self.encoder_outputs
        #encoder_last_state = self.encoder_last_state
        #encoder_inputs_length = self.encoder_inputs_length

        # To use BeamSearchDecoder, encoder_outputs, encoder_last_state, encoder_inputs_length 
        # needs to be tiled so that: [batch_size, .., ..] -> [batch_size x beam_width, .., ..]
        if use_beamsearch_decode:
            print ("use beamsearch decoding..")
            encoder_outputs = seq2seq.tile_batch(
                encoder_outputs, multiplier=beam_width)
            encoder_last_state = nest.map_structure(
                lambda s: seq2seq.tile_batch(s, beam_width), encoder_last_state)
            #encoder_last_state = seq2seq.tile_batch(encoder_last_state,multiplier=beam_width)
            encoder_inputs_length = seq2seq.tile_batch(
                encoder_inputs_length, multiplier=beam_width)

        # Building attention mechanism: Default Bahdanau
        # 'Bahdanau' style attention: https://arxiv.org/abs/1409.0473
        if attention_type.lower() == 'bahdanau':
            attention_mechanism = attention_wrapper.BahdanauAttention(
                num_units=hidden_units, memory=encoder_outputs,
                memory_sequence_length=encoder_inputs_length,) 
        # 'Luong' style attention: https://arxiv.org/abs/1508.04025
        elif attention_type.lower() == 'luong':
            attention_mechanism = attention_wrapper.LuongAttention(
                num_units=hidden_units, memory=encoder_outputs, 
                memory_sequence_length=encoder_inputs_length,)
        else: raise ValueError('the attention_type is undefined:{}'.format(attention_type))
 
        # Building decoder_cell
        decoder_cell_list = [
            self.build_single_cell(keep_prob_placehoder,rnn_cell_type = rnn_cell_type,
                hidden_units = hidden_units,use_dropout = use_dropout, use_residual = use_residual) for i in range(num_layers)]
        decoder_initial_state = encoder_last_state

        def attn_decoder_input_fn(inputs, attention):
            if not attn_input_feeding:
                return inputs

            # Essential when use_residual=True
            _input_layer = Dense(hidden_units, dtype=self.dtype,
                                 name='attn_input_feeding')
            return _input_layer(array_ops.concat([inputs, attention], -1))

        # AttentionWrapper wraps RNNCell with the attention_mechanism
        # Note: We implement Attention mechanism only on the top decoder layer
        decoder_cell_list[-1] = attention_wrapper.AttentionWrapper(
            cell=decoder_cell_list[-1],
            attention_mechanism=attention_mechanism,
            attention_layer_size=hidden_units,
            cell_input_fn=attn_decoder_input_fn,
            initial_cell_state=encoder_last_state[-1],
            alignment_history=False,
            name='Attention_Wrapper')

        # To be compatible with AttentionWrapper, the encoder last state
        # of the top layer should be converted into the AttentionWrapperState form
        # We can easily do this by calling AttentionWrapper.zero_state

        # Also if beamsearch decoding is used, the batch_size argument in .zero_state
        # should be ${decoder_beam_width} times to the origianl batch_size
        times_batch_size = tf.shape(encoder_outputs)[0] # already times beam_width if use_beamsearch
        #times_batch_size = batch_size if not use_beamsearch_decode \
        #             else batch_size * beam_width
        initial_state = [state for state in encoder_last_state]

        initial_state[-1] = decoder_cell_list[-1].zero_state(
          batch_size=times_batch_size, dtype=self.dtype)#.clone(cell_state=encoder_last_state)
        decoder_initial_state = tuple(initial_state)

        return MultiRNNCell(decoder_cell_list), decoder_initial_state

    @layer
    def attn_decoder(self, input, name,attention_type,attn_input_feeding,
            rnn_cell_type,hidden_units,beam_width, use_beamsearch_decode,
            use_residual, use_dropout, num_layers,num_decoder_symbols,embedding_size):
        print("building decoder and attention..")
        (keep_prob_placehoder, (encoder_outputs,encoder_last_state),
                encoder_inputs_length,decoder_inputs,decoder_inputs_length) = input
        with tf.variable_scope(name) as scope:
            # Building decoder_cell and decoder_initial_state , when training use_beamsearch_decode is false
            if self.mode=='train': 
                use_beamsearch_decode = False
                beam_width = 1
            decoder_cell, decoder_initial_state  = \
            self.build_decoder_cell( keep_prob_placehoder,
                encoder_outputs,encoder_last_state,encoder_inputs_length,
                attention_type=attention_type,attn_input_feeding=attn_input_feeding,
                rnn_cell_type=rnn_cell_type,hidden_units=hidden_units,beam_width=beam_width,
                use_beamsearch_decode = use_beamsearch_decode,use_dropout = use_dropout,
                use_residual = use_residual, num_layers=num_layers)

            # Initialize decoder embeddings to have variance=1.
            sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=self.dtype)
             
            decoder_embeddings = tf.get_variable(name='embedding',
                shape=[num_decoder_symbols, embedding_size],
                initializer=initializer, dtype=self.dtype)

            # Input projection layer to feed embedded inputs to the cell
            # ** Essential when use_residual=True to match input/output dims
            input_layer = Dense(hidden_units, dtype=self.dtype, name='input_projection')

            # Output projection layer to convert cell_outputs to logits
            output_layer = Dense(num_decoder_symbols, name='output_projection')
            batch_size = tf.shape(encoder_inputs_length)[0]

            decoder_start_token = tf.ones(
                shape=[batch_size, 1], dtype=tf.int32) * cfg.GO_TOKEN
            decoder_end_token = tf.ones(
                shape=[batch_size, 1], dtype=tf.int32) * cfg.EOS_TOKEN

            # decoder_inputs_train: [batch_size , max_time_steps + 1]
            # insert _GO symbol in front of each decoder input
            decoder_inputs_train = tf.concat([decoder_start_token,
                                                  decoder_inputs], axis=1)

            # decoder_inputs_length_train: [batch_size]
            decoder_inputs_length_train = decoder_inputs_length + 1

            # decoder_targets_train: [batch_size, max_time_steps + 1]
            # insert EOS symbol at the end of each decoder input
            decoder_targets_train = tf.concat([decoder_inputs,
                                                   decoder_end_token], axis=1)


            if self.mode == 'train':
                # decoder_inputs_embedded: [batch_size, max_time_step + 1, embedding_size]
                decoder_inputs_embedded = tf.nn.embedding_lookup(
                    params=decoder_embeddings, ids=decoder_inputs_train)
               
                # Embedded inputs having gone through input projection layer
                decoder_inputs_embedded = input_layer(decoder_inputs_embedded)

                # Helper to feed inputs for training: read inputs from dense ground truth vectors
                training_helper = seq2seq.TrainingHelper(inputs=decoder_inputs_embedded,
                                                   sequence_length=decoder_inputs_length_train,
                                                   time_major=False,
                                                   name='training_helper')

                training_decoder = seq2seq.BasicDecoder(cell=decoder_cell,
                                                   helper=training_helper,
                                                   initial_state=decoder_initial_state,
                                                   output_layer=output_layer)
                                                   #output_layer=None)
                    
                # Maximum decoder time_steps in current batch
                max_decoder_length = tf.reduce_max(decoder_inputs_length_train)

                # decoder_outputs_train: BasicDecoderOutput
                #                        namedtuple(rnn_outputs, sample_id)
                # decoder_outputs_train.rnn_output: [batch_size, max_time_step + 1, num_decoder_symbols] if output_time_major=False
                #                                   [max_time_step + 1, batch_size, num_decoder_symbols] if output_time_major=True
                # decoder_outputs_train.sample_id: [batch_size], tf.int32
                (decoder_outputs_train, decoder_last_state_train, 
                 decoder_outputs_length_train) = (seq2seq.dynamic_decode(
                    decoder=training_decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=max_decoder_length))
                 
                # More efficient to do the projection on the batch-time-concatenated tensor
                # logits_train: [batch_size, max_time_step + 1, num_decoder_symbols]
                # self.decoder_logits_train = output_layer(self.decoder_outputs_train.rnn_output)
                decoder_logits_train = tf.identity(decoder_outputs_train.rnn_output) 
                # Use argmax to extract decoder symbols to emit
                #decoder_pred_train = tf.argmax(decoder_logits_train, axis=-1,
                #                                    name='decoder_pred_train')

                # masks: masking for valid and padded time steps, [batch_size, max_time_step + 1]
                masks = tf.sequence_mask(lengths=decoder_inputs_length_train, 
                                         maxlen=max_decoder_length, dtype=self.dtype, name='masks')

                # Computes per word average cross-entropy over a batch
                # Internally calls 'nn_ops.sparse_softmax_cross_entropy_with_logits' by default
                loss = seq2seq.sequence_loss(logits=decoder_logits_train, 
                                                  targets=decoder_targets_train,
                                                  weights=masks,
                                                  average_across_timesteps=True,
                                                  average_across_batch=True,)
                #return loss
                # Training summary for the current batch_loss
                #tf.summary.scalar('loss', self.loss)

                # Contruct graphs for minimizing loss
                #self.init_optimizer()

            #elif self.mode == 'decode':
            if True:
        
                # Start_tokens: [batch_size,] `int32` vector
                start_tokens = tf.ones([batch_size,], tf.int32) * cfg.GO_TOKEN
                end_token = cfg.EOS_TOKEN

                def embed_and_input_proj(inputs):
                    return input_layer(tf.nn.embedding_lookup(decoder_embeddings, inputs))
                    
                if not use_beamsearch_decode:
                    # Helper to feed inputs for greedy decoding: uses the argmax of the output
                    decoding_helper = seq2seq.GreedyEmbeddingHelper(start_tokens=start_tokens,
                                                                    end_token=end_token,
                                                                    embedding=embed_and_input_proj)
                    # Basic decoder performs greedy decoding at each time step
                    print("building greedy decoder..")
                    inference_decoder = seq2seq.BasicDecoder(cell=decoder_cell,
                                                             helper=decoding_helper,
                                                             initial_state=decoder_initial_state,
                                                             output_layer=output_layer)
                else:
                    # Beamsearch is used to approximately find the most likely translation
                    print("building beamsearch decoder..")
                    inference_decoder = beam_search_decoder.BeamSearchDecoder(cell=decoder_cell,
                                                               embedding=embed_and_input_proj,
                                                               start_tokens=start_tokens,
                                                               end_token=end_token,
                                                               initial_state=decoder_initial_state,
                                                               beam_width=beam_width,
                                                               output_layer=output_layer,)
                # For GreedyDecoder, return
                # decoder_outputs_decode: BasicDecoderOutput instance
                #                         namedtuple(rnn_outputs, sample_id)
                # decoder_outputs_decode.rnn_output: [batch_size, max_time_step, num_decoder_symbols] 	if output_time_major=False
                #                                    [max_time_step, batch_size, num_decoder_symbols] 	if output_time_major=True
                # decoder_outputs_decode.sample_id: [batch_size, max_time_step], tf.int32		if output_time_major=False
                #                                   [max_time_step, batch_size], tf.int32               if output_time_major=True 
                
                # For BeamSearchDecoder, return
                # decoder_outputs_decode: FinalBeamSearchDecoderOutput instance
                #                         namedtuple(predicted_ids, beam_search_decoder_output)
                # decoder_outputs_decode.predicted_ids: [batch_size, max_time_step, beam_width] if output_time_major=False
                #                                       [max_time_step, batch_size, beam_width] if output_time_major=True
                # decoder_outputs_decode.beam_search_decoder_output: BeamSearchDecoderOutput instance
                #                                                    namedtuple(scores, predicted_ids, parent_ids)

                max_decode_step = cfg.MAX_DECODE_STEP
                (decoder_outputs_decode, decoder_last_state_decode,
                 decoder_outputs_length_decode) = (seq2seq.dynamic_decode(
                    decoder=inference_decoder,
                    output_time_major=False,
                    #impute_finished=True,	# error occurs
                    maximum_iterations=max_decode_step))

                if not use_beamsearch_decode:
                    # decoder_outputs_decode.sample_id: [batch_size, max_time_step]
                    # Or use argmax to find decoder symbols to emit:
                    # self.decoder_pred_decode = tf.argmax(self.decoder_outputs_decode.rnn_output,
                    #                                      axis=-1, name='decoder_pred_decode')

                    # Here, we use expand_dims to be compatible with the result of the beamsearch decoder
                    # decoder_pred_decode: [batch_size, max_time_step, 1] (output_major=False)
                    decoder_pred_decode = tf.expand_dims(decoder_outputs_decode.sample_id, -1)

                else:
                    # Use beam search to approximately find the most likely translation
                    # decoder_pred_decode: [batch_size, max_time_step, beam_width] (output_major=False)
                    decoder_pred_decode = decoder_outputs_decode.predicted_ids
                if self.mode == 'decode': loss = tf.constant(0,dtype=self.dtype) # for the decode part
                return loss,tf.squeeze(tf.split(decoder_pred_decode,num_or_size_splits=beam_width,axis=-1)[0],axis=-1)
                #return loss, decoder_pred_decode


    @layer
    def decoder(self, input, name, hidden_units = 1024,trainable=True ):
        encoder_outputs,encoder_last_state = input
        with tf.variable_scope(name) as scope:
            batch_size = tf.shape(encoder_outputs)[0]
            encoder_outputs = tf.reshape(encoder_outputs,[-1, hidden_units ])
            init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
            # init_weights = tf.contrib.layers.xavier_initializer()
            # init_weights = tf.truncated_normal_initializer(stddev=0.1)
            init_biases = tf.constant_initializer(0.0)
            W = self.make_var('weights', [hidden_units, cfg.NCLASSES], init_weights, trainable, \
                              regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            b = self.make_var('biases', [cfg.NCLASSES], init_biases, trainable)
            logits = tf.matmul(encoder_outputs,W)+b
            logits = tf.reshape(logits,[batch_size,-1,cfg.NCLASSES])
            logits = tf.transpose(logits,(1,0,2))
            return logits

    def build_loss(self):
        time_step_batch = self.get_output('time_step_len')
        #logits_batch = self.get_output('loss/logits')
        labels = self.get_output('labels')
        label_len = self.get_output('labels_len')

        #ctc_loss = warpctc_tensorflow.ctc(activations=logits_batch,flat_labels=labels,
        #                                       label_lengths=label_len,input_lengths=time_step_batch)
        loss,dense_decoded = self.get_output('loss_decode_res')
        #loss = self.get_output('loss')
        loss = tf.reduce_mean(loss)
        #decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits_batch, time_step_batch, merge_repeated=True)
        #dense_decoded = tf.cast(tf.sparse_tensor_to_dense(decoded[0], default_value=0), tf.int32)

        # add regularizer
        if cfg.TRAIN.WEIGHT_DECAY > 0:
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss = tf.add_n(regularization_losses) + loss

        return loss,dense_decoded
