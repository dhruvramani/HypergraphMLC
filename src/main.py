import os
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from dataset import DataSet

batch_size, feature_dim, label_dim = 300, 200, 100
def dot(x, y, sparse=False):
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

def hypergraph(np_labels):
    incidence = np_labels.T
    weightDiag = sp.eye(np_labels.shape[0])
    edgesDiag = sp.eye((np_labels.shape[0]))
    vertexDiag = sp.eye((np_labels.shape[1]))
    sum_0 = sp.csr_matrx(sp.csr_matrix.sum(labels, axis=0))
    sum_1 = sp.csr_matrix(tf.reduce_sum(labels, axis=1))
    edgesDiag = sp.csr_matrix.multiply(sum_0, edgesDiag)
    vertexDiag = sp.csr_matrix.multiply(sum_1, vertexDiag)
    return incidence, weightDiag, edgesDiag, vertexDiag

def get_lap(labels):
    H, W, De, Dv = hypergraph(labels)
    # I - Dv^(-1/2).H.De^(-1).Ht.Dv^(-1/2)
    L = sp.eye(labels.shape[0]) - sp.csr_matrix.sqrt(sp.linalg.inv(Dv)).dot(H.dot(sp.linalg.inv(De).dot((H.T).dot(sp.csr_matrix.sqrt(sp.linalg.inv(Dv))))))
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
        for epoch in range(1000):
            el, c = 0.0, 0
            for x_train, y_train in get_batch("train"):
                laplacian = get_lap(y_train)
                pl, _ = sess.run([loss, train], feed_dict={X: x_train, Y: y_train, laps: laplacian})
                el += pl
                c += 1
                print("Epoch #{} Loss : {}".format(epoch, pl), end='\r')
            print("Epoch #{} Loss : {}".format(epoch, el/c))
