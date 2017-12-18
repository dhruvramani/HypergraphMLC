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

def get_sparse_props(spmatrix):
    indices = list()
    prop = sp.find(spmatrix)
    for i in range(prop[0].shape[0]):
        indices.append(np.array([prop[1][i], prop[0][i]]))
    indices = np.array(indices)
    values = prop[2]
    shape = spmatrix.shape
    return indices, values, shape

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
    X_indices = tf.placeholder(tf.int64, name='X_indices', shape=None)
    X_data = tf.placeholder(tf.float32, name='X_data', shape=None)
    X_shape = tf.placeholder(tf.int64, name='X_shape', shape=None)

    Y_indices = tf.placeholder(tf.int64, name='Y_indices', shape=None)
    Y_data = tf.placeholder(tf.float32, name='Y_data', shape=None)
    Y_shape = tf.placeholder(tf.int64, name='Y_shape', shape=None)

    L_indices = tf.placeholder(tf.int64, name='L_indices', shape=None)
    L_data = tf.placeholder(tf.float32, name='L_data', shape=None)
    L_shape = tf.placeholder(tf.int64, name='L_shape', shape=None)

    X = tf.SparseTensor(indices=X_indices, values=X_data, shape=X_shape)
    Y = tf.SparseTensor(indices=Y_indices, values=Y_data, shape=Y_shape)
    laps = tf.SparseTensor(indices=L_indices, values=L_data, shape=L_shape)

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

    tf.scalar_summary("loss1", loss1)
    tf.scalar_summary("loss2", loss2)
    tf.scalar_summary("loss3", loss3)

    loss = loss1 + loss2 + loss3
    tf.scalar_summary("loss", loss)
    train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter("./tensorboard", sess.graph)
        for epoch in range(1000):
            el, c = 0.0, 0
            dataobj = DataSet("./data/delicious/delicious-train", batch_size)
            for x_train, y_train, dummy in dataobj.next_batch("train"):
                laplacian = get_lap(y_train)
                x_props, y_props, l_props = get_sparse_props(x_train), get_sparse_props(y_train), get_sparse_props(laplacian)
                feed = {X_indices : x_props[0], X_data : x_props[1], X_shape : x_props[2], Y_indices : y_props[0], Y_data : y_props[1], Y_shape : y_props[2], L_indices : l_props[0], L_data : l_props[1], L_shape : l_props[2]}
                pl, _, summ = sess.run([loss, train, merged], feed_dict=feed)
                el += pl
                c += 1
                print("Epoch #{} Loss : {}".format(epoch, pl), end='\r')
            saver.save(sess, './checkpoint/model.ckpt', global_step=c)
            writer.add_summary(summ, c)
            print("Epoch #{} Loss : {}".format(epoch, el/c))