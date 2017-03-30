import os,sys
#import config
import numpy as np
import tensorflow as tf
import random
import cv2,time
from tensorflow.python.client import device_lib
import utils

FLAGS=utils.FLAGS
#26*2 + 10 digit + blank + space
num_classes=utils.num_classes
#num_classes=10+1+1
num_train_samples = utils.num_train_samples # 10000

num_features=utils.num_features
num_batches_per_epoch = int(num_train_samples/FLAGS.batch_size) # 10000/32


graph = tf.Graph()
#with tf.get_default_graph()._kernel_label_map({'CTCLoss':'WarpCTC'}):
with graph.as_default():
    # e.g: log filter bank or MFCC features
    # Has size [batch_size, max_stepsize, num_features], but the
    # batch_size and max_stepsize can vary along each step
    inputs = tf.placeholder(tf.float32, [None, None, num_features])
    
    # Here we use sparse_placeholder that will generate a
    # SparseTensor required by ctc_loss op.
    labels = tf.sparse_placeholder(tf.int32)
    
    # 1d array of size [batch_size]
    seq_len = tf.placeholder(tf.int32, [None])
    
    # Defining the cell
    # Can be:
    #   tf.nn.rnn_cell.RNNCell
    #   tf.nn.rnn_cell.GRUCell
    cells=[tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, state_is_tuple=True) for i in range(FLAGS.num_layers)]
    #cell1 = tf.contrib.rnn.core_rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
    
    # Stacking rnn cells
    stack = tf.contrib.rnn.MultiRNNCell(cells ,
                                        state_is_tuple=True)
    
    # The second output is the last state and we will no use that
    outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)
    
    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]
    
    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, FLAGS.num_hidden])
    
    # Truncated normal with mean 0 and stdev=0.1
    # Tip: Try another initialization
    # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
    W = tf.Variable(tf.truncated_normal([FLAGS.num_hidden,
                                         num_classes],
                                        stddev=0.1),name='W')
    # Zero initialization
    # Tip: Is tf.zeros_initializer the same?
    b = tf.Variable(tf.constant(0., shape=[num_classes],name='b'))
   
    # Doing the affine projection
    logits = tf.matmul(outputs, W) + b
   
    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_s, -1, num_classes])
   
    # Time major
    logits = tf.transpose(logits, (1, 0, 2))
   
    global_step = tf.Variable(0,trainable=False)
   
    loss = tf.nn.ctc_loss(labels=labels,inputs=logits, sequence_length=seq_len)
    cost = tf.reduce_mean(loss)
   
    learning_rate=tf.train.exponential_decay(FLAGS.initial_learning_rate,global_step,
            FLAGS.decay_steps,FLAGS.decay_rate,staircase=True)
    #optimizer = tf.train.MomentumOptimizer(initial_learning_rate,
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
            momentum=FLAGS.momentum).minimize(cost,global_step=global_step)
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
    #        beta1=FLAGS.beta1,beta2=FLAGS.beta2).minimize(loss,global_step=global_step)
   
    # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len,merge_repeated=False)
    #decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len,merge_repeated=False)
   
    # Inaccuracy: label error rate
    lerr = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), labels))

def train():
    train_feeder=utils.DataIterator(data_dir='./train1/')
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        print('=============================begin training=============================')
        for cur_epoch in range(FLAGS.num_epochs):
            shuffle_idx=np.random.permutation(num_train_samples)
            train_cost = train_err=0
            start = time.time()
            for cur_batch in range(num_batches_per_epoch):
                indexs = [shuffle_idx[i%num_train_samples] for i in range(cur_batch*FLAGS.batch_size,(cur_batch+1)*FLAGS.batch_size)]
                batch_inputs,batch_seq_len,batch_labels=train_feeder.input_index_generate_batch(indexs)
                feed={inputs: batch_inputs,
                        labels:batch_labels,
                        seq_len:batch_seq_len}
                batch_cost, _,the_err,d,lr = sess.run([cost,optimizer,lerr,decoded[0],learning_rate],feed)
                train_cost+=batch_cost*FLAGS.batch_size
                #err,d,lr=sess.run(lerr,decoded[0],learning_rate,feed_dict=feed)
                train_err+=the_err*FLAGS.batch_size
                #train_err+=sess.run(lerr,feed_dict=feed)*FLAGS.batch_size
                dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=sess)
            for i, seq in enumerate(dense_decoded):
                seq = [s for s in seq if s != -1]
                print('Sequence %d' % i, end='  ')
                print('Original:\n%s' % train_feeder.the_label(indexs)[i])
                print('Decoded:\n%s' % seq)
            train_cost/=num_train_samples
            train_err/=num_train_samples
            log = "Epoch {}/{}, train_cost = {:.3f}, train_err = {:.3f}, time = {:.3f},lr={:.7f}"
            print(log.format(cur_epoch+1,FLAGS.num_epochs,train_cost,train_err,time.time()-start,lr))



        #saver=tf.train.Saver()

        #train_writer=tf.summary.FileWriter(FLAGS.log_dir+'/train',sess.graph)
        #try:
        #    while not coord.should_stop():
        #        start_time=time.time()
        

if __name__ == '__main__':
    train()

