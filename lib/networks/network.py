import numpy as np
import tensorflow as tf

from lib.lstm.config import cfg
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
        #img = tf.squeeze(img,axis=3)
        if img_shape:img =tf.reshape(img,shape = img_shape )
        with tf.variable_scope(name) as scope:
            #stack = tf.contrib.rnn.MultiRNNCell([cell,cell1] , state_is_tuple=True)
            lstm_fw_cell = tf.contrib.rnn.LSTMCell(num_hids//2,state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.LSTMCell(num_hids//2,state_is_tuple=True)

            output,_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,img,img_len,dtype=tf.float32)
            # output_bw_reverse = tf.reverse_sequence(output[1],img_len,seq_axis=1)
            output = tf.concat(output,axis=2)

            #stack_cell = tf.contrib.rnn.MultiRNNCell(
            #    [tf.contrib.rnn.LSTMCell(num_hids, state_is_tuple=True) for _ in range(num_layers)],
            #    state_is_tuple=True)
            #lstm_out,last_state = tf.nn.dynamic_rnn(stack_cell,output,img_len,dtype=tf.float32)
            lstm_out = output
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
            stack_cell = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.LSTMCell(num_hids, state_is_tuple=True) for _ in range(num_layers)],
                state_is_tuple=True)
            lstm_out,last_state = tf.nn.dynamic_rnn(stack_cell,img,img_len,dtype=tf.float32)
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
    def reshape_squeeze_layer(self, input, d, name):
        #N,H,W,C-> N,H*W,C
        input_shape = tf.shape(input)
        return tf.reshape(input, \
                          [input_shape[0], \
                           input_shape[1]*input_shape[2], \
                           int(d)])

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


    def build_loss(self):
        time_step_batch = self.get_output('time_step_len')
        logits_batch = self.get_output('logits')
        labels = self.get_output('labels')
        label_len = self.get_output('labels_len')

        ctc_loss = warpctc_tensorflow.ctc(activations=logits_batch,flat_labels=labels,
                                               label_lengths=label_len,input_lengths=time_step_batch)
        loss = tf.reduce_mean(ctc_loss)
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits_batch, time_step_batch, merge_repeated=True)
        dense_decoded = tf.cast(tf.sparse_tensor_to_dense(decoded[0], default_value=0), tf.int32)

        # add regularizer
        if cfg.TRAIN.WEIGHT_DECAY > 0:
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss = tf.add_n(regularization_losses) + loss

        return loss,dense_decoded
