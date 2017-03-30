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
            im = cv2.imread(image_name,0).astype(np.float32)/255.
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





        #saver=tf.train.Saver()

        #train_writer=tf.summary.FileWriter(FLAGS.log_dir+'/train',sess.graph)
        #try:
        #    while not coord.should_stop():
        #        start_time=time.time()
        

