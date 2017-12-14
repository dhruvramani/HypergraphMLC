import argparse
import numpy as np
from datetime import datetime

class Parser(object):  #
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--label_loss", default=1, help="Supervised learning weightage", type=self.str2bool)
        parser.add_argument("--")
        parser.add_argument("--save_embeddings", default=False, help="save embeddings for unsupervised model", type=self.str2bool)

        parser.add_argument("--max_depth", default=2, help="Maximum path depth", type=int)
        parser.add_argument("--dims", default='128,128,128,128', help="Dimensions of hidden layers: comma separated")

        parser.add_argument("--filter_sizes", default="3,4,5", help="Rows of Filters: comma separated")
        parser.add_argument("--sequence_length", default=59, help="Sentence Length (Row of TO-CONV)")

        parser.add_argument("--bias", default=False, type=self.str2bool)
        parser.add_argument("--sparse_features", default=True, help="For current datasets - manually set in config.py", type=self.str2bool)

        # Dataset Details
        parser.add_argument("--dataset", default='cora', help="Dataset to evluate | Check Datasets folder",
                            choices=['cora', 'citeseer', 'wiki', 'amazon', 'facebook', 'cora_multi', 'movielens',
                                    'ppi_sg', 'blogcatalog', 'genes_fn', 'mlgene', 'ppi_gs', 'reddit', 'reddit_ind'])

        # NN Hyper parameters
        parser.add_argument("--dropout", default=0.8, help="Dropout Keeprob", type=float)
        parser.add_argument("--batch_size", default=128, help="Batch size", type=int)
        parser.add_argument("--lr", default=1e-2, help="Learning rate", type=float)
        parser.add_argument("--l2", default=1e-6, help="L2 loss", type=float)
        parser.add_argument("--opt", default='adam', help="Optimizer type", choices=['adam', 'sgd', 'rmsprop'])

        # Training parameters
        parser.add_argument("--retrain", default=False, type=self.str2bool, help="Retrain flag")
        parser.add_argument("--gpu", default=0, help="GPU BUS ID ", type=int)
        parser.add_argument("--verbose", default=0, help="Verbose mode", type=int, choices=[0, 1, 2])
        parser.add_argument("--save_model", default=False, type=self.str2bool)

        parser.add_argument("--max_epoch", default=1, help="Maximum outer epoch", type=int)
        parser.add_argument("--lu", default=0.8, help="Label update rate", type=float)

        parser.add_argument("--pat", default=15, help="Patience", type=int)
        parser.add_argument("--pat_inc", default=2, help="Patience Increase", type=int)
        parser.add_argument("--pat_improve", default=.9999, help="Improvement threshold for patience", type=float)
        parser.add_argument("--save_after", default=30, help="Save after epochs", type=int)
        parser.add_argument("--val_freq", default=1, help="Validation frequency", type=int)
        parser.add_argument("--summaries", default=True, help="Save summaries after each epoch", type=self.str2bool)

        now = datetime.now()
        timestamp = str(now.month) + '|' + str(now.day) + '|' + str(now.hour) + ':' + str(now.minute) + ':' + str(
            now.second)
        parser.add_argument("--timestamp", default=timestamp, help="Timestamp to prefix experiment dumps")
        parser.add_argument("--folder_suffix", default='Default', help="folder name suffix")

        # TODO Load saved model and saved argparse
        self.parser = parser


    def str2bool(self, text):
        if text == 'True':
            arg = True
        elif text == 'False':
            arg = False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        return arg


    def get_parser(self):
        return self.parser
