import os
import numpy as np
import tensorflow as tf
import scipy.sparse as sp

batch_size, feature_dim, label_dim = 300, 200, 100
'''
def dense_layer(input, input_dim, output_dim, act=tf.nn.relu, dropout=0.8):
    W = tf.Variable(tf.random_normal(shape=[input_dim, output_dim]))
    b = tf.Variable(tf.random_normal(shape=[output_dim]))
    return act(tf.bias_add(dot(W, input)))

def hypergraph(labels):
    shape = labels.shape
    incidence = labels.T
    weightDiag = sp.eye(shape[0])

    #edgesDiag = sp.eye(shape[0])
    #vertexDiag = sp.eye(shape[1])
    #sum_0 = sp.expand_dims(sp.sum(labels, axis=0), 0)
    #sum_1 = tf.expand_dims(tf.reduce_sum(labels, axis=1), 0)
    #edgesDiag = tf.multiply(sum_0, edgesDiag)
    #vertexDiag = tf.multiply(sum_1, vertexDiag)
'''
def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

def ce_loss(predictions, labels):
    cross_entropy = tf.add(tf.multiply(tf.log(1e-10 + tf.nn.sigmoid(predictions)), labels),
                    tf.multiply(tf.log(1e-10 + (1 - tf.nn.sigmoid(predictions))), (1 - labels)))
    loss = -1 * tf.reduce_mean(tf.reduce_sum(cross_entropy, 1), name='xentropy_mean')
    return loss

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

def get_lap(labels):
    H, W, De, Dv = hypergraph(labels)
    L = tf.eye(labels.get_shape().as_list()[0]) - dot(tf.sqrt(tf.reciprocal(Dv)), dot(H, dot(tf.matrix_inverse(De), dot(tf.transpose(H), tf.sqrt(tf.reciprocal(Dv))))))
    return L

def model():
    X = tf.placeholder(tf.float32, shape=[batch_size, feature_dim])
    Y = tf.placeholder(tf.float32, shape=[batch_size, label_dim])
    laps = tf.placeholder(tf.float32, shape=[batch_size, batch_size]) 

    Wx1 = tf.Variable(tf.random_normal(shape=[feature_dim, 300]))
    bx1 = tf.Variable(tf.random_normal(shape=[300]))
    Wx2 = tf.Variable(tf.random_normal(shape=[300, 100]))
    bx2 = tf.Variable(tf.random_normal(shape=[100])) 

    Wy1 = tf.Variable(tf.random_normal(shape=[label_dim, 300]))
    by1 = tf.Variable(tf.random_normal(shape=[300]))
    Wy2 = tf.Variable(tf.random_normal(shape=[300, 100]))
    by2 = tf.Variable(tf.random_normal(shape=[100]))

    Wh1 = tf.Variable(tf.random_normal(shape=[100, 300]))
    bh1 = tf.Variable(tf.random_normal(shape=[300]))
    Wh2 = tf.Variable(tf.random_normal(shape=[300, label_dim]))
    bh2 = tf.Variable(tf.random_normal(shape=[label_dim]))

    act = tf.nn.relu
    hx1 = act(dot(X, Wx1) + bx1)
    hxe = act(dot(hx1, Wx2) + bx2)
    
    hy1 = act(dot(Y, Wy1) + by1)
    hye = act(dot(hy1, Wy2) + by2)
   
    hhx1 = act(dot(hxe, Wh1) + bh1)
    hhx2 = dot(hhx1, Wh2) + bh2

    hhy1 = act(dot(hye, Wh1) + bh1)
    hhy2 = dot(hhy1, Wh2) + bh2

    loss1 = ce_loss(hhx2, Y) + ce_loss(hhy2, Y)
    loss2 = dot(dot(tf.transpose(hxe), laps), hxe)
    loss3 = dot(dot(tf.transpose(hye), laps), hye)

    loss = loss1 + loss2 + loss3
    train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(1000):
            x_train, y_train = get_batch("train")
            laplacian = get_lap(y_train)
            pl, _ = sess.run([loss, train], feed_dict={X: x_train, Y: y_train, laps: laplacian})
            print("Loss : {}".format(pl), end='\r')
    