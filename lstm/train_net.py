import argparse
import pprint
import numpy as np
import pdb
import sys
import os.path

this_dir = os.path.dirname(__file__)
sys.path.insert(0, this_dir + '/..')

from lib.lstm.train import train_net
from lib.lstm.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_log_dir
from lib.networks.factory import get_network
from easydict import EasyDict as edict
import matplotlib

def parse_args():
    parser = argparse.ArgumentParser(description='Train a lstm network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=700000, type=int)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--pre_train', dest='pre_train',
                        help='pre trained model',
                        default=None, type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--restore', dest='restore',
                        help='restore or not',
                        default=0, type=int)

    if len(sys.argv) == 1:
        parser.print_help()

    args = parser.parse_args()
    return args

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPU_ID)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    # imgdb = edict({'path':'data/lstm_voc/pascal_augmented_train.tfrecords','name':'pascal_augmentted'})
    output_network_name=args.network_name.split('_')[-1]
    imgdb = edict({'path':'./data/train_4_6.tfrecords','name':'lstm_'+output_network_name,
                   'val_path':'./data/val.tfrecords' })

    output_dir = get_output_dir(imgdb, None)
    log_dir = get_log_dir(imgdb)
    print(('Output will be saved to `{:s}`'.format(output_dir)))
    print(('Logs will be saved to `{:s}`'.format(log_dir)))

    device_name = '/gpu:{:d}'.format(args.gpu_id)
    print(device_name)

    network = get_network(args.network_name)
    print(('Use network `{:s}` in training'.format(args.network_name)))

    train_net(network, imgdb,
              pre_train=args.pre_train,
              output_dir=output_dir,
              log_dir=log_dir,
              max_iters=args.max_iters,
              restore=bool(int(args.restore)))
