import tensorflow as tf
import sys
from yaml import dump
from os import path
from src.utils import utils
import numpy as np
import importlib


class Config(object):
    def __init__(self, args):

        # SET UP PATHS
        self.paths = dict()
        self.paths['root'] = '../'

        self.paths['datasets'] = path.join(self.paths['root'], 'Datasets')
        self.paths['experiments'] = path.join(self.paths['root'], 'Experiments')
        self.dataset_name = args.dataset
        self.paths['experiment'] = path.join(self.paths['experiments'], args.timestamp, self.dataset_name, args.folder_suffix)
        # Parse training percentages and folds

        suffix = self.paths['experiment']
        path_prefix = self.paths['experiment']
        self.paths['logs' + suffix] = path.join(path_prefix, 'Logs/')
        self.paths['ckpt' + suffix] = path.join(path_prefix, 'Checkpoints/')
        self.paths['embed' + suffix] = path.join(path_prefix, 'Embeddings/')
        self.paths['results' + suffix] = path.join(path_prefix, 'Results/')

        # Create directories
        for (key, val) in self.paths.items():
            if key not in ['root', 'experiments', 'datasets']:
                utils.create_directory_tree(str.split(val, sep='/')[:-1])

        dump(args.__dict__, open(path.join(self.paths['experiment'], 'args.yaml'), 'w'), default_flow_style=False, explicit_start=True)

        self.paths['data'] = path.join(self.paths['datasets'], self.dataset_name)
        self.paths['labels'] = path.join(path.join(self.paths['data'], 'labels.npy'))
        self.paths['features'] = path.join(path.join(self.paths['data'], 'features.npy'))
        self.paths['adjmat'] = path.join(path.join(self.paths['data'], 'adjmat.mat'))

        # -------------------------------------------------------------------------------------------------------------

        # Hidden dimensions
        self.dims = list(map(int, args.dims.split(',')[:args.max_depth]))
        if len(self.dims) < args.max_depth:
            sys.exit('#Hidden dimensions should match the max depth')

        # Propogation Depth
        self.max_depth = args.max_depth

        # GPU
        self.gpu = args.gpu

        # Weighed cross entropy loss
        self.wce = args.wce

        # Retrain
        self.retrain = args.retrain

        # Metrics to compute
        self.metric_keys = ['accuracy', 'micro_f1', 'macro_f1', 'bae']

        # Batch size
        if args.batch_size == -1:
            self.queue_capacity = 1
        else:
            self.queue_capacity = 5
        self.batch_size = args.batch_size

        # Dropouts
        self.dropout = args.dropout

        # Number of steps to run trainer
        self.max_epoch = args.max_epoch

        # ConvText Properties
        self.filter_sizes = [int(i) for i in args.filter_sizes.split(',')]
        self.num_filters = len(filter_sizes)
        
        self.sequence_length = args.sequence_length

        # Save summaries
        self.summaries = args.summaries

        # Validation frequence
        self.val_epochs_freq = args.val_freq  # 1
        # Model save frequency
        self.save_epochs_after = args.save_after  # 0

        # early stopping hyper parametrs
        self.patience = args.pat  # look as this many epochs regardless
        self.patience_increase = args.pat_inc  # wait this much longer when a new best is found
        self.improvement_threshold = args.pat_improve  # a relative improvement of this much is considered significant

        self.learning_rate = args.lr
        self.label_update_rate = args.lu

        # optimizer
        self.l2 = args.l2
        self.l2 = args.l2
        if args.opt == 'adam':
            self.opt = tf.train.AdamOptimizer
        elif args.opt == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer
        elif args.opt == 'sgd':
            self.opt = tf.train.GradientDescentOptimizer
        else:
            raise ValueError('Undefined type of optmizer')

        # -------------------------------------------------------------------------------------------------------------

        # Sparse Feature settings
        self.sparse_features = args.sparse_features
        if not self.sparse_features and self.dataset_name in ['cora', 'citeseer', 'amazon', 'facebook', 'cora_multi', 'movielens',
                                    'ppi_sg', 'blogcatalog', 'genes_fn', 'mlgene', 'ppi_gs']:
            self.sparse_features = True
            print('Sparse Features turned on forcibly!')
        elif self.dataset_name in ['wiki', 'reddit']:
            self.sparse_features = False

        # Node features
        self.features = ['x', 'h']

        # Loss terms
        self.loss = {}
        self.loss['label'] = args.label_loss
        self.loss['l2'] = args.l2

        if args.shared_weights == 1:
            self.shared_weights = True
        else:
            self.shared_weights = False
        self.bias = args.bias

        self.add_labels = False
        if self.max_outer_epochs > 1:
            self.add_labels = True

        self.save_model = args.save_model