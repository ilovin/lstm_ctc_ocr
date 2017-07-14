import os
import tensorflow as tf
import numpy as np
from easydict import EasyDict as edict
from lib.networks.factory import get_network
from lib.fcn.config import get_output_dir,cfg_from_file

class Convert(object):
    def __init__(self, sess, network, model_dir,out_path,model):
        self.net = network
        self.model_dir = model_dir
        self.out_path=out_path
        self.model=model
        self.saver = tf.train.Saver(max_to_keep=100)

    def conver2npy(self,sess):
        global_step = tf.Variable(0, trainable=False)
        local_vars_init_op = tf.local_variables_initializer()
        global_vars_init_op = tf.global_variables_initializer()
        combined_op = tf.group(local_vars_init_op, global_vars_init_op)
        sess.run(combined_op)

        try:
            self.saver.restore(sess, tf.train.latest_checkpoint(self.model_dir))
            sess.run(global_step.assign(0))
            dic=dict()
            pri_keys=['conv1_1','conv1_2','conv2_1','conv2_2',
                  'conv3_1','conv3_2','conv3_3',
                  'conv4_1','conv4_2','conv4_3',
                  'conv5_1','conv5_2','conv5_3']
            if self.model==32:
                keys=pri_keys+['fc6','fc7','fc8']
            elif self.model==16:
                keys=pri_keys+['fc6','fc7','fc8','pool4_fc']
            elif self.model==8:
                keys=pri_keys+['fc6','fc7','fc8','pool4_fc','pool3_fc']
            for key in keys:
                with tf.variable_scope(key, reuse=True):
                    dic[key] = dict()
                    for subkey in ['weights','biases']:
                        try:
                            var = tf.get_variable(subkey)
                            data=sess.run(var)
                            dic[key][subkey]=data

                            print("save model " + subkey + " to " + key)
                        except ValueError:
                            print("fail to convert")
            np.save(self.out_path, dic)
        except:
            raise Exception('Check your model')


def convert_ckpt2npy(network, model_dir,out_path,model):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allocator_type = 'BFC'
    with tf.Session(config=config) as sess:
        ct = Convert(sess, network,model_dir,out_path,model)
        ct.conver2npy(sess)
        print('done converting')




if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    #修改此处
    output_network_name = '32s'

    cfg_from_file('./fcn/fcn_nlpr.yml')
    imgdb = edict({'path': './data/train.tfrecords', 'name': 'FCN_' + output_network_name})
    model_dir = get_output_dir(imgdb, None)
    network = get_network('VGGnet_'+output_network_name)
    out_path='./data/'+output_network_name
    convert_ckpt2npy(network,model_dir=model_dir,out_path=out_path,model=int(output_network_name[:-1]))
