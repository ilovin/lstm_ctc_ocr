import os,sys
#import config
import numpy as np
import tensorflow as tf
import random
import cv2,time
import logging,datetime
from tensorflow.python.client import device_lib
from tensorflow.python.client import timeline
import utils

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 

FLAGS=utils.FLAGS
#26*2 + 10 digit + blank + space
num_classes=utils.num_classes

num_features=utils.num_features

logger = logging.getLogger('Traing for ocr using LSTM+CTC')
logger.setLevel(logging.INFO)
#with tf.get_default_graph()._kernel_label_map({'CTCLoss':'WarpCTC'}):
#with tf.device('/gpu:1'):
class Graph(object):
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # e.g: log filter bank or MFCC features
            # Has size [batch_size, max_stepsize, num_features], but the
            # batch_size and max_stepsize can vary along each step
            self.inputs = tf.placeholder(tf.float32, [None, None, num_features])
            
            # Here we use sparse_placeholder that will generate a
            # SparseTensor required by ctc_loss op.
            self.labels = tf.sparse_placeholder(tf.int32)
            
            # 1d array of size [batch_size]
            self.seq_len = tf.placeholder(tf.int32, [None])
            
            # Defining the cell
            # Can be:
            #   tf.nn.rnn_cell.RNNCell
            #   tf.nn.rnn_cell.GRUCell
            #cell = tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, state_is_tuple=True)
            #cell = tf.contrib.rnn.DropoutWrapper(cell = cell,output_keep_prob=0.8)
            #
            #cell1 = tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, state_is_tuple=True)
            #cell1 = tf.contrib.rnn.DropoutWrapper(cell = cell1,output_keep_prob=0.8)
            # Stacking rnn cells
            #stack = tf.contrib.rnn.MultiRNNCell([cell,cell1] , state_is_tuple=True)
            stack = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(FLAGS.num_hidden,state_is_tuple=True) for _ in range(FLAGS.num_layers)] , state_is_tuple=True)
            
            # The second output is the last state and we will no use that
            outputs, _ = tf.nn.dynamic_rnn(stack, self.inputs, self.seq_len, dtype=tf.float32)
            
            shape = tf.shape(self.inputs)
            batch_s, max_timesteps = shape[0], shape[1]
            
            # Reshaping to apply the same weights over the timesteps
            outputs = tf.reshape(outputs, [-1, FLAGS.num_hidden])
            
            # Truncated normal with mean 0 and stdev=0.1
            # Tip: Try another initialization
            # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
            W = tf.Variable(tf.truncated_normal([FLAGS.num_hidden,
                                                 num_classes],
                                                stddev=0.1,dtype=tf.float32),name='W')
            # Zero initialization
            # Tip: Is tf.zeros_initializer the same?
            b = tf.Variable(tf.constant(0., dtype = tf.float32,shape=[num_classes],name='b'))
           
            # Doing the affine projection
            logits = tf.matmul(outputs, W) + b
           
            # Reshaping back to the original shape
            logits = tf.reshape(logits, [batch_s, -1, num_classes])
           
            # Time major
            logits = tf.transpose(logits, (1, 0, 2))
           
            self.global_step = tf.Variable(0,trainable=False)
           
        
            self.loss = tf.nn.ctc_loss(labels=self.labels,inputs=logits, sequence_length=self.seq_len)
            self.cost = tf.reduce_mean(self.loss)
        
            #learning_rate=tf.train.exponential_decay(FLAGS.initial_learning_rate,
            #        global_step, 
            #        FLAGS.decay_steps,
            #        FLAGS.decay_rate,staircase=True)
           
            #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
            #        momentum=FLAGS.momentum,use_nesterov=True).minimize(cost,global_step=global_step)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.initial_learning_rate,
                    beta1=FLAGS.beta1,beta2=FLAGS.beta2).minimize(self.loss,global_step=self.global_step)
           
            # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
            # (it's slower but you'll get better results)
            #decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len,merge_repeated=False)
            self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(logits, self.seq_len,merge_repeated=False)
           
            # Inaccuracy: label error rate
            self.lerr = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.labels))
        
            tf.summary.scalar('cost',self.cost)
            tf.summary.scalar('lerr',self.lerr)
            self.merged_summay = tf.summary.merge_all()

def train():
    g = Graph()
    with g.graph.as_default():
        print('loading train data, please wait---------------------',end=' ')
        train_feeder=utils.DataIterator(data_dir='../train/')
        print('get image: ',train_feeder.size)

        print('loading validation data, please wait---------------------',end=' ')
        val_feeder=utils.DataIterator(data_dir='../test/')
        print('get image: ',val_feeder.size)

    num_train_samples = train_feeder.size # 12800
    num_batches_per_epoch = int(num_train_samples/FLAGS.batch_size) # example: 12800/64

    config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=False)
    with tf.Session(graph=g.graph,config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=100)
        train_writer=tf.summary.FileWriter(FLAGS.log_dir+'/train',sess.graph)
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                # the global_step will restore sa well
                saver.restore(sess,ckpt)
                print('restore from the checkpoint{0}'.format(ckpt))

        print('=============================begin training=============================')
        # the cuda trace
        #run_metadata = tf.RunMetadata()
        #trace_file = open('timeline.ctf.json','w')
        val_inputs,val_seq_len,val_labels=val_feeder.input_index_generate_batch()
        val_feed={g.inputs: val_inputs,
                  g.labels: val_labels,
                 g.seq_len: val_seq_len}
        for cur_epoch in range(FLAGS.num_epochs):
            shuffle_idx=np.random.permutation(num_train_samples)
            train_cost = train_err=0
            start = time.time()
            batch_time = time.time()
            #the tracing part
            for cur_batch in range(num_batches_per_epoch):
                if (cur_batch+1)%100==0:
                    print('batch',cur_batch,': time',time.time()-batch_time)
                batch_time = time.time()
                indexs = [shuffle_idx[i%num_train_samples] for i in range(cur_batch*FLAGS.batch_size,(cur_batch+1)*FLAGS.batch_size)]
                batch_inputs,batch_seq_len,batch_labels=train_feeder.input_index_generate_batch(indexs)
                feed={g.inputs: batch_inputs,
                        g.labels:batch_labels,
                        g.seq_len:batch_seq_len}
                #_,batch_cost, the_err,d,lr,train_summary,step = sess.run([optimizer,cost,lerr,decoded[0],learning_rate,merged_summay,global_step],feed)
                #_,batch_cost, the_err,d,lr,step = sess.run([optimizer,cost,lerr,decoded[0],learning_rate,global_step],feed)
                #the_err,d,lr = sess.run([lerr,decoded[0],learning_rate])

                # if summary is needed
                #batch_cost,step,train_summary,_ = sess.run([cost,global_step,merged_summay,optimizer],feed)
                batch_cost,step,_ = sess.run([g.cost,g.global_step,g.optimizer],feed)
                #calculate the cost
                train_cost+=batch_cost*FLAGS.batch_size
                ## the tracing part
                #_,batch_cost,the_err,step,lr,d = sess.run([optimizer,cost,lerr,
                #    global_step,learning_rate,decoded[0]],feed)
                    #options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                    #run_metadata=run_metadata)
                #trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                #race_file.write(trace.generate_chrome_trace_format())
                #trace_file.close()

                #train_writer.add_summary(train_summary,step)

                # save the checkpoint
                if step%FLAGS.save_steps == 1:
                    if not os.path.isdir(FLAGS.checkpoint_dir):
                        os.mkdir(FLAGS.checkpoint_dir)
                    logger.info('save the checkpoint of{0}',format(step))
                    saver.save(sess,os.path.join(FLAGS.checkpoint_dir,'ocr-model'),global_step=step)
                #train_err+=the_err*FLAGS.batch_size
            d,lastbatch_err = sess.run([g.decoded[0],g.lerr],val_feed)
            dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=sess)
            # print the decode result
            acc = utils.accuracy_calculation(val_feeder.labels,dense_decoded,ignore_value=-1,isPrint=True)
            train_cost/=num_train_samples
            #train_err/=num_train_samples
            now = datetime.datetime.now()
            log = "{}-{} {}:{}:{} Epoch {}/{}, accuracy = {:.3f},train_cost = {:.3f}, lastbatch_err = {:.3f}, time = {:.3f}"
            print(log.format(now.month,now.day,now.hour,now.minute,now.second,
                cur_epoch+1,FLAGS.num_epochs,acc,train_cost,lastbatch_err,time.time()-start))
        

if __name__ == '__main__':
    train()

