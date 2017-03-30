import os,sys
#import config
import numpy as np
import tensorflow as tf
import random
import cv2,time
from tensorflow.python.client import device_lib

#26*2 + 10 digit + blank + space
num_classes=26+26+10+1+1
#num_classes=10+1+1
num_train_samples = 10000

num_features=60
image_width=160
image_height=60

SPACE_INDEX=0
SPACE_TOKEN='<space>'

tf.app.flags.DEFINE_string('num_layers', 1, 'number of layer')
tf.app.flags.DEFINE_string('num_hidden', 32, 'number of hidde')
tf.app.flags.DEFINE_string('num_epochs', 10000, 'total epochs')
tf.app.flags.DEFINE_string('batch_size', 32, 'the batch_size')
tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')
tf.app.flags.DEFINE_string('decay_steps', 2000, 'the lr decay_step')
tf.app.flags.DEFINE_string('decay_rate', 0.97, 'the lr decay rate')
tf.app.flags.DEFINE_string('beta1', 0.9, 'parameter of adam optimizer beta1')
tf.app.flags.DEFINE_string('beta2', 0.999, 'adam parameter beta2')
tf.app.flags.DEFINE_string('initial_learning_rate', 1e-3, 'inital lr')
tf.app.flags.DEFINE_string('momentum', 0.9, 'the momentum')
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')
FLAGS=tf.app.flags.FLAGS

num_batches_per_epoch = int(num_train_samples/FLAGS.batch_size)

maps  = {}
maps_value = 11
for i in range(10):
    maps[str(i)]=i+1
for char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
    maps[char] = maps_value
    maps_value += 1
def get_label(buf):
    global maps
    ret = np.zeros(len(buf))
    for i in range(len(buf)):
        ret[i] = maps[buf[i]]
    return ret

def sparse_tuple_from_label(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape

def pad_input_sequences(sequences, maxlen=None, dtype=np.float32,
                  padding='post', truncating='post', value=0.):
    '''Pads each sequence to the same length: the length of the longest
    sequence.
        If maxlen is provided, any sequence longer than maxlen is truncated to
        maxlen. Truncation happens off either the beginning or the end
        (default) of the sequence. Supports post-padding (default) and
        pre-padding.
        Args:
            sequences: list of lists where each element is a sequence
            maxlen: int, maximum length
            dtype: type to cast the resulting sequence.
            padding: 'pre' or 'post', pad either before or after each sequence.
            truncating: 'pre' or 'post', remove values from sequences larger
            than maxlen either in the beginning or in the end of the sequence
            value: float, value to pad the sequences to the desired value.
        Returns
            x: numpy array with dimensions (number_of_sequences, maxlen)
            lengths: numpy array with the original sequence lengths
    '''
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x, lengths
class DataIterator:
    def __init__(self, data_dir):
        # Set FLAGS.charset_size to a small value if available computation power is limited.
        #truncate_path = data_dir + ('%05d' % FLAGS.charset_size)
        #print(truncate_path)
        self.image_names = []
        for root, sub_folder, file_list in os.walk(data_dir):
            self.image_names += [os.path.join(root, file_path) for file_path in file_list]
        random.shuffle(self.image_names)
        self.labels=[]
        for idx,file_name in enumerate(self.image_names):
            code = file_name.split('/')[2].split('_')[1].split('.')[0]
            code = [SPACE_INDEX if code == SPACE_TOKEN else maps[c] for c in list(code)]
            self.labels.append(code)

    @property
    def size(self):
        return len(self.labels)

    def the_label(self,indexs):
        labs=[]
        for i in indexs:
            labs.append(self.labels[i])
        return labs

    @staticmethod
    def data_augmentation(images):
        if FLAGS.random_flip_up_down:
            images = tf.image.random_flip_up_down(images)
        if FLAGS.random_brightness:
            images = tf.image.random_brightness(images, max_delta=0.3)
        if FLAGS.random_contrast:
            images = tf.image.random_contrast(images, 0.8, 1.2)
        return images

    def input_index_generate_batch(self,index,aug=False):
        image_batch=[]
        label_batch=[]
        for i in index:
            image_name = self.image_names[i]
            im = cv2.imread(image_name)[:,:,0].astype(np.float32)/255.
            im = cv2.resize(im,(image_width,image_height))
            # transpose to (160*60) and the step shall be 160
            im = im.transpose()
            # in this way, each row is a feature vector
            image_batch.append(np.array(list(im)))
            label_batch.append(np.array(self.labels[i]))
        image_batch=np.array(image_batch)
        #print(image_batch.shape)
        batch_inputs,batch_seq_len = pad_input_sequences(image_batch)
        batch_labels = sparse_tuple_from_label(label_batch)
        return batch_inputs,batch_seq_len,batch_labels



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
    train_feeder=DataIterator(data_dir='./train1/')
    with tf.Session(graph=graph) as sess:
        #train_images,train_labels=train_feeder.input_pipeline(batch_size=FLAGS.batch_size,aug=False)
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

