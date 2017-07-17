# Important: We are using PIL to read .png files later.
# This was done on purpose to read indexed png files
# in a special way -- only indexes and not map the indexes
# to actual rgb values. This is specific to PASCAL VOC
# dataset data. If you don't want thit type of behaviour
# consider using skimage.io.imread()
from PIL import Image
import numpy as np
import skimage.io as io
import tensorflow as tf
import os,re
from lib.lstm.config import cfg,get_encode_decode_dict

charset = cfg.CHARSET
encode_maps , decode_maps = get_encode_decode_dict()
# Helper functions for defining tf types
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto,
    e.g, sentence in list of ints
    """
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto,
    e.g, sentence in list of bytes
    """
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def write_image_annotation_pairs_to_tfrecord(img_path, tfrecords_filename):
    """Writes given image/annotation pairs to the tfrecords file.
    The function reads each image/annotation pair given img_path
    of image and respective annotation and writes it to the tfrecord
    file.
    Parameters
    ----------
    img_path : img_path
    tfrecords_filename : string
        Tfrecords filename to write the image/annotation pairs
    """
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    maxLen = cfg.MAX_CHAR_LEN

    for root,subfolder,fileList in os.walk(img_path):
        for fname in fileList:
            fname = os.path.join(root,fname)
            img = np.array(Image.open(fname))
            code = re.match(r'.*\/[0-9]+_(.*)(_1)?\..*', fname).group(1)
            code = [cfg.SPACE_INDEX if code == cfg.SPACE_TOKEN else encode_maps[c] for c in list(code)]
            aligned_code = code[:]
            while len(aligned_code)<maxLen:aligned_code.append(0)
            # Unomment this one when working with surgical data

            # The reason to store image sizes was demonstrated
            # in the previous example -- we have to know sizes
            # of images to later read raw serialized string,
            # convert to 1d array and convert to respective
            # shape that image used to have.
            height = img.shape[0]
            width = img.shape[1]
            time_step = cfg.IMG_SHAPE[0]#160

            img_raw = img.tostring()

            context = tf.train.Features(feature={
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'time_step': _int64_feature(time_step),
                'label_len': _int64_feature(len(code)),
                'image_raw': _bytes_feature(img_raw)
                })
            featureLists = tf.train.FeatureLists(feature_list={
                'label': _int64_feature_list(aligned_code)
            })

            sequence_example = tf.train.SequenceExample(
                context=context,feature_lists =featureLists
            )

            writer.write(sequence_example.SerializeToString())
            # writer.write(example.SerializeToString())

        writer.close()
        print('Done')


def read_image_annotation_pairs_from_tfrecord(tfrecords_filename):
    """Return image/annotation pairs from the tfrecords file.
    The function reads the tfrecords file and returns image
    and respective annotation matrices pairs.
    Parameters
    ----------
    tfrecords_filename : string
        filename of .tfrecords file to read from
    
    Returns
    -------
    image_annotation_pairs : array of tuples (img, annotation)
        The image and annotation that were read from the file
    """
    
    image_annotation_pairs = []

    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

    for string_record in record_iterator:

        example = tf.train.Example()
        example.ParseFromString(string_record)

        height = int(example.features.feature['height']
                                     .int64_list
                                     .value[0])

        width = int(example.features.feature['width']
                                    .int64_list
                                    .value[0])

        img_string = (example.features.feature['image_raw']
                                      .bytes_list
                                      .value[0])

        annotation_string = (example.features.feature['mask_raw']
                                    .bytes_list
                                    .value[0])

        img_1d = np.fromstring(img_string, dtype=np.uint8)
        img = img_1d.reshape((height, width, -1))

        annotation_1d = np.fromstring(annotation_string, dtype=np.uint8)

        # Annotations don't have depth (3rd dimension)
        # TODO: check if it works for other datasets
        annotation = annotation_1d.reshape((height, width))

        image_annotation_pairs.append((img, annotation))
    
    return image_annotation_pairs


def read_tfrecord_and_decode_into_image_annotation_pair_tensors(tfrecord_filenames_queue):
    """Return image/annotation tensors that are created by reading tfrecord file.
    The function accepts tfrecord filenames queue as an input which is usually
    can be created using tf.train.string_input_producer() where filename
    is specified with desired number of epochs. This function takes queue
    produced by aforemention tf.train.string_input_producer() and defines
    tensors converted from raw binary representations into
    reshaped image/annotation tensors.
    Parameters
    ----------
    tfrecord_filenames_queue : tfrecord filename queue
        String queue object from tf.train.string_input_producer()
    
    Returns
    -------
    image, annotation : tuple of tf.int32 (image, annotation)
        Tuple of image/annotation tensors
    """
    
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(tfrecord_filenames_queue)

    features,sequence_features = tf.parse_single_sequence_example( serialized_example,
        context_features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'time_step': tf.FixedLenFeature([], tf.int64),
            'label_len': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string), },
        sequence_features={
            'label': tf.FixedLenSequenceFeature([], tf.int64),})
    
    image = tf.decode_raw(features['image_raw'], tf.uint8)

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    label_len = tf.cast(features['label_len'], tf.int32)
    label = tf.cast(sequence_features['label'],tf.int32)
    label = tf.reshape(label,[cfg.MAX_CHAR_LEN])
    #image_shape = tf.pack([height, width, 3])
    image_shape = tf.parallel_stack([height, width, 3])
    image = tf.reshape(image,image_shape)

    img_size = cfg.IMG_SHAPE #160,60
    time_step = tf.constant(cfg.TIME_STEP,tf.int32)

    if cfg.NCHANNELS==1: image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize_images(image,size=(img_size[1],img_size[0]),method=tf.image.ResizeMethod.BILINEAR)
    image = tf.transpose(image,perm=[1,0,2])
    image = tf.cast(tf.reshape(image,[img_size[0],cfg.NUM_FEATURES]),dtype=tf.float32)/255.

    # The last dimension was added because
    # the tf.resize_image_with_crop_or_pad() accepts tensors
    # that have depth. We need resize and crop later.
    # TODO: See if it is necessary and probably remove third
    # dimension
    #annotation_shape = tf.pack([height, width, 1])
    # image = tf.reshape(image, image_shape)

    return image, label,label_len,time_step

def wrtie_test(img_path ,tfrecords_filename = None):
    write_image_annotation_pairs_to_tfrecord(img_path=img_path,tfrecords_filename=tfrecords_filename)
def read_test(tfrecords_fiename=None):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allocator_type = 'BFC'
    with tf.Session(config=config) as sess:
        filename_queue = tf.train.string_input_producer([tfrecords_fiename], num_epochs=1)
        image,label,label_len,time_step= read_tfrecord_and_decode_into_image_annotation_pair_tensors(filename_queue)
        image_batch, label_batch, label_len_batch,time_step_batch = tf.train.shuffle_batch([image,label,label_len,time_step],
                                                                                           batch_size=2,
                                                                                           capacity=500,
                                                                                           num_threads=2,
                                                                                           min_after_dequeue=100)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop():
                # get one batch
                img_batch, l_batch, l_len_batch,t_s_batch = sess.run([image_batch, label_batch, label_len_batch,time_step_batch] )
                label = []
                for l in l_batch:
                    while l[-1] == 0: l=l[:-1]
                    label.extend(l)
                print(l_batch)
                # Subtract the mean pixel value from each pixel
        except tf.errors.OutOfRangeError:
            print('finish')
        finally:
            coord.request_stop()
        coord.join(threads)


if __name__=='__main__':
    wrtie_test(img_path='/home/amax/Documents/code/lstm_train/lstm_ctc/data/train_4_6',tfrecords_filename='./data/train_4_6.tfrecords')
    # wrtie_test(img_path='./data/val',tfrecords_filename='./data/val.tfrecords')
    # read_test(tfrecords_fiename='./data/val.tfrecords')
