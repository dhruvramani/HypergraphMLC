import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from os import path, mkdir
from sklearn import preprocessing
import shutil
import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.io as sio

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

def sparse_relu(x):
    # TODO: Ensure gradients are computed properly, especially because of using x-x for 0
    # minus_one = tf.constant(-1, shape=[[1]], dtype=tf.float32)
    x = tf.sparse_add(x, x, thresh=0) #Max(0,x)
    return x

def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

def check_n_create(dir_path, overwrite=False):
    if not path.exists(dir_path):
        mkdir(dir_path)
    else:
        if overwrite:
            shutil.rmtree(dir_path)
            mkdir(dir_path)

def create_directory_tree(dir_path):
    for i in range(len(dir_path)):
        check_n_create(path.join(*(dir_path[:i + 1])))

def remove_directory(dir_path):
    shutil.rmtree(dir_path, ignore_errors=True)

def get_part_symm_laplacian(adj, drows, dcols):
    with np.errstate(divide='ignore', invalid='ignore'):
        drow_inv_sqrt = np.power(drows, -0.5).flatten()
        drow_inv_sqrt[np.isinf(drow_inv_sqrt)] = 0.
        drow_inv_sqrt = sp.diags(drow_inv_sqrt)

        if drows.shape != dcols.shape:
            dcol_inv_sqrt = np.power(dcols, -0.5).flatten()
            dcol_inv_sqrt[np.isinf(dcol_inv_sqrt)] = 0.
            dcol_inv_sqrt = sp.diags(dcol_inv_sqrt)
        else:
            dcol_inv_sqrt = drow_inv_sqrt

    return adj.dot(dcol_inv_sqrt).transpose().dot(drow_inv_sqrt)

def hypergraph(labels):
    shape = labels.get_shape.as_list()
    incidence = tf.transpose(labels)
    weightDiag = tf.eye(shape[0])
    edgesDiag = tf.eye(shape[0])
    vertexDiag = tf.eye(shape[1])
    sum_0 = tf.expand_dims(tf.reduce_sum(labels, axis=0), 0)
    sum_1 = tf.expand_dims(tf.reduce_sum(labels, axis=1), 0)
    edgesDiag = tf.multiply(sum_0, edgesDiag)
    vertexDiag = tf.multiply(sum_1, vertexDiag)
    return incidence, weightDiag, edgesDiag, vertexDiag

def get_symm_normalized_laplcian(labels):
    H, W, De, Dv = hypergraph(labels)
    L = tf.eye(labels.get_shape().as_list()[0]) - tf.matmul(tf.sqrt(tf.reciprocal(Dv)), tf.matmul(H, tf.matmul(tf.matrix_inverse(De), tf.matmul(tf.transpose(H), tf.sqrt(tf.reciprocal(Dv))))))
    return L

def get_symm_laplacian_Term2(adjmat, degrees):
    degrees = tf.expand_dims(1/tf.sqrt(degrees), axis=1)
    laplacian = adjmat.__mul__(degrees)
    degrees = tf.transpose(degrees, [1, 0])
    laplacian = laplacian.__mul__(degrees)
    return laplacian

def construct_feed_dict(features, support, labels, labels_mask, placeholders, lr=0, dropouts=(0,0,0)):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['learning_rate']: lr})

    feed_dict.update({placeholders['dropout']: dropouts[0]})
    feed_dict.update({placeholders['dropout_conv']: dropouts[1]})
    feed_dict.update({placeholders['support']: drop_connect(support, dropouts[2])})
    # feed_dict.update({placeholders['support']: support})

    return feed_dict

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

'''
#   NOT NEEEDED CURRENTLY
def preprocess_features(features, degrees):
    features = preprocessing.normalize(features, norm='l1')
    return features

def preprocess_features2(features, degrees):
    features = features.todense()
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(features)
    features = scaler.transform(features)
    return features

def add_degree(features, degrees):
    log_degrees = np.log(degrees+1)
    maxi = np.max(log_degrees)
    log_degrees /= maxi
    return sp.hstack([features, sp.csr_matrix(log_degrees)])


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def norm_deg(adj):
    """Symmetrically normalizing degree matrix."""
    adj = sp.coo_matrix(adj)
    degree = np.array(adj.sum(1))
    with np.errstate(divide='ignore', invalid='ignore'):
        d_inv_sqrt = np.power(degree, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    return degree, np.expand_dims(d_inv_sqrt, axis=1)

def asym_normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.  # Handles -Inf, Inf, NaN
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_I    = adj + sp.eye(adj.shape[0])
    # adj_I    = adj
    # adj_I = normalize_adj(adj + sp.eye(adj.shape[0]))
    #adj_I = normalize_adj(adj) + sp.eye(adj.shape[0])
    # return sparse_to_tuple(adj_I)
    return sp.csr_matrix(adj_I, dtype=np.bool)

def preprocess_adj2(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    # adj_I    = adj + sp.eye(adj.shape[0])
    # adj_I    = adj
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    #adj_normalized = normalize_adj(adj) + sp.eye(adj.shape[0])
    return adj_normalized


def print_Att(message, A1, A2):
    spc = sp.csr_matrix
    A1 = spc((A1.values, (A1.indices[:,0], A1.indices[:,1])), A1.dense_shape)
    A2 = spc((A2.values, (A2.indices[:, 0], A2.indices[:, 1])), A2.dense_shape)

    print(message + "\t A1: {}::{} || A2: {}::{}".format(np.mean(spc.min(A1, -1)), np.mean(spc.max(A1, -1)),
                                                         np.mean(spc.min(A2, -1)), np.mean(spc.max(A2, -1))))


def drop_connect(adj, drop=0):
    """
    Randomly drop few edges (to make learning more robust)
    """
    indices, values, shape = adj
    size = len(values)
    perm = np.random.permutation(size)[int(drop*size):]
    indices, values = indices[perm], values[perm]

    return indices, values, shape


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def get_wce(labels, training, validation, flag):
    with np.errstate(divide='ignore', invalid='ignore'):
        if flag:
            valid = np.logical_or(training, validation)
            tot = np.sum(labels[valid], axis=0)
            wce = 1 / (len(tot) * (tot * 1.0 / np.sum(tot)))
        else:
            wce = np.ones(shape[1])

        wce[np.isinf(wce)] = 0
        wce[np.isnan(wce)] = 0
    return wce


def csc_row_set_nz_to_val(csc, row, value=0):
    """Set all nonzero elements (elements currently in the sparsity pattern)
    to the given value. Useful to set to 0 mostly.
    """
    if not isinstance(csc, sp.csc_matrix):
        raise ValueError('Matrix given must be of CSR format.')
    csc.data[csc.indptr[row]:csc.indptr[row+1]] = value
    return csc
'''