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
num_train_samples = 128000

num_features=60
image_width=160
image_height=60

SPACE_INDEX=0
SPACE_TOKEN=''

tf.app.flags.DEFINE_boolean('restore', True, 'whether to restore from the latest checkpoint')

tf.app.flags.DEFINE_integer('num_layers', 2, 'number of layer')
tf.app.flags.DEFINE_integer('num_hidden', 100, 'number of hidden')
tf.app.flags.DEFINE_integer('num_epochs', 10000, 'maximum epochs')
tf.app.flags.DEFINE_integer('batch_size', 64, 'the batch_size')
tf.app.flags.DEFINE_integer('save_steps', 1000, 'the step to save checkpoint')

tf.app.flags.DEFINE_float('initial_learning_rate', 1e-3, 'inital lr')

tf.app.flags.DEFINE_float('decay_rate', 0.9, 'the lr decay rate')
tf.app.flags.DEFINE_float('beta1', 0.9, 'parameter of adam optimizer beta1')
tf.app.flags.DEFINE_float('beta2', 0.999, 'adam parameter beta2')

tf.app.flags.DEFINE_integer('decay_steps', 10000, 'the lr decay_step for momentum optimizer')
tf.app.flags.DEFINE_float('momentum', 0.9, 'the momentum')

tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')
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
        self.image_names = []
        self.image = []
        self.labels=[]
        for root, sub_folder, file_list in os.walk(data_dir):
            for file_path in file_list:
                image_name = os.path.join(root,file_path)
                self.image_names.append(image_name)
                im = cv2.imread(image_name,0).astype(np.float32)/255.
                im = cv2.resize(im,(image_width,image_height))
                # transpose to (160*60) and the step shall be 160
                # in this way, each row is a feature vector
                im = im.swapaxes(0,1)
                self.image.append(np.array(im))
                #image is named as ./<folder>/00000_abcd.png
                code = image_name.split('/')[2].split('_')[1].split('.')[0]
                code = [SPACE_INDEX if code == SPACE_TOKEN else maps[c] for c in list(code)]
                self.labels.append(code)
                print(image_name,' ',code)
        #random.shuffle(self.image_names)

    @property
    def size(self):
        return len(self.labels)

    def the_label(self,indexs):
        labels=[]
        for i in indexs:
            labels.append(self.labels[i])
        return labels

    #@staticmethod
    #def data_augmentation(images):
    #    if FLAGS.random_flip_up_down:
    #        images = tf.image.random_flip_up_down(images)
    #    if FLAGS.random_brightness:
    #        images = tf.image.random_brightness(images, max_delta=0.3)
    #    if FLAGS.random_contrast:
    #        images = tf.image.random_contrast(images, 0.8, 1.2)
    #    return images

    def input_index_generate_batch(self,index=None):
        if index:
            image_batch=[self.image[i] for i in index]
            label_batch=[self.labels[i] for i in index]
        else:
            # get the whole data as input
            image_batch=self.image
            label_batch=self.labels

        def get_input_lens(sequences):
            lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)
            return sequences,lengths
        batch_inputs,batch_seq_len = get_input_lens(np.array(image_batch))
        batch_labels = sparse_tuple_from_label(label_batch)
        return batch_inputs,batch_seq_len,batch_labels

    def input_index_generate_batch_warp(self,index):
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
        batch_labels,batch_labels_len = get_label_and_lens(label_batch)
        sparse_labels = sparse_tuple_from_label(label_batch)
        return batch_inputs,batch_seq_len,batch_labels,batch_labels_len,sparse_labels

def accuracy_calculation(original_seq,decoded_seq,ignore_value=-1,isPrint = True):
    if  len(original_seq)!=len(decoded_seq):
        print('original lengths is different from the decoded_seq,please check again')
        return 0
    count = 0
    for i,origin_label in enumerate(original_seq):
        decoded_label  = [j for j in decoded_seq[i] if j!=ignore_value]
        if isPrint:
            print('seq{0:4d}: origin: {1} decoded:{2}'.format(i,origin_label,decoded_label))
        if origin_label == decoded_label: count+=1
    return count*1.0/len(original_seq)

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

def get_label_and_lens(sequences,dtype=np.int32):
    '''
    Args:
        sequences:a list of lists of dtype where each element is a sequence
    Returns:
        A tuple with(flat_labels,per_lebel_len_list)
    '''
    flat_labels = []
    labels_len=[]

    for n, seq in enumerate(sequences):
        flat_labels.extend(seq)
        labels_len.append(len(seq))

    flat_labels = np.asarray(flat_labels, dtype=dtype)
    labels_len = np.asarray(labels_len, dtype=dtype)
    return flat_labels,labels_len
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

