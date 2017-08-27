import os
import os.path as osp
import numpy as np
from time import strftime, localtime
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

# Default GPU device id
__C.GPU_ID = 1
# region proposal network (RPN) or not
__C.POOL_SCALE = 2
__C.IMG_SHAPE = [180,60]
__C.MAX_CHAR_LEN = 6
__C.CHARSET = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
__C.NCLASSES = len(__C.CHARSET)+2
__C.MIN_LEN = 4
__C.MAX_LEN = 6
__C.FONT = 'fonts/Ubuntu-M.ttf'
__C.NCHANNELS = 1
__C.NUM_FEATURES=__C.IMG_SHAPE[1]*__C.NCHANNELS
__C.TIME_STEP = __C.IMG_SHAPE[0]//__C.POOL_SCALE

__C.NET_NAME = 'lstm'
__C.TRAIN = edict()
# Adam, Momentum, RMS
__C.TRAIN.SOLVER = 'Adam'
#__C.TRAIN.SOLVER = 'Momentum'
# __C.TRAIN.SOLVER = 'RMS'
# learning rate
__C.TRAIN.WEIGHT_DECAY = 0.0005
__C.TRAIN.LEARNING_RATE = 0.01
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.GAMMA = 0.1
__C.TRAIN.STEPSIZE = 50000
__C.TRAIN.DISPLAY = 10
__C.TRAIN.LOG_IMAGE_ITERS = 100
__C.TRAIN.NUM_EPOCHS = 2000

__C.TRAIN.NUM_HID = 128
__C.TRAIN.NUM_LAYERS = 2
__C.TRAIN.BATCH_SIZE = 32

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 5000
__C.TRAIN.SNAPSHOT_PREFIX = 'lstm'
__C.TRAIN.SNAPSHOT_INFIX = ''

__C.VAL = edict()
__C.VAL.VAL_STEP = 500
__C.VAL.NUM_EPOCHS = 1000
__C.VAL.BATCH_SIZE = 128
__C.VAL.PRINT_NUM = 5

__C.RNG_SEED = 3

__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
__C.TEST = edict()
__C.EXP_DIR = 'default'
__C.LOG_DIR = 'default'

__C.SPACE_INDEX = 0
__C.SPACE_TOKEN = ''
def get_encode_decode_dict():
    encode_maps = {}
    decode_maps = {}
    for i, char in enumerate(__C.CHARSET, 1):
        encode_maps[char] = i
        decode_maps[i] = char
    encode_maps[__C.SPACE_TOKEN] = __C.SPACE_INDEX
    decode_maps[__C.SPACE_INDEX] = __C.SPACE_TOKEN
    return encode_maps,decode_maps


def get_output_dir(imdb, weights_filename):
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR))
    if weights_filename is not None:
        outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir

def get_log_dir(imdb):
    log_dir = osp.abspath(\
        osp.join(__C.ROOT_DIR, 'logs', __C.LOG_DIR, imdb.name, strftime("%Y-%m-%d-%H-%M-%S", localtime())))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def _merge_a_into_b(a, b):
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
