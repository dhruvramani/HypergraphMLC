import os
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
import scipy.sparse.linalg as la
from dataset import DataSet

no_epoch, batch_size, feature_dim, label_dim = 1000, 300, 500, 983
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
        indices.append(np.array([prop[0][i], prop[1][i]]))
    indices = np.array(indices)
    values = prop[2]
    shape = spmatrix.shape
    return indices, values, shape

def hypergraph(np_labels):
    incidence = np_labels.T
    weightDiag = sp.eye(np_labels.shape[0])
    edgesDiag = sp.eye((np_labels.shape[0]))
    vertexDiag = sp.eye((np_labels.shape[1]))
    sum_0 = sp.csr_matrix(sp.csr_matrix.sum(np_labels, axis=1))
    sum_1 = sp.csr_matrix(sp.csr_matrix.sum(np_labels, axis=0))
    print(sum_0.shape, edgesDiag.shape)
    edgesDiag = sp.csr_matrix.multiply(sum_0, edgesDiag)
    vertexDiag = sp.csr_matrix.multiply(sum_1, vertexDiag)
    return incidence, weightDiag, edgesDiag, vertexDiag

def get_lap(labels):
    H, W, De, Dv = hypergraph(labels)
    # I - Dv^(-1/2).H.De^(-1).Ht.Dv^(-1/2)
    L = sp.eye(labels.shape[0]) - sp.csr_matrix.sqrt(la.inv(Dv)).dot(H.dot(la.inv(De).dot((H.T).dot(sp.csr_matrix.sqrt(la.inv(Dv))))))
    return L

def model():
    X_indices = tf.placeholder(tf.int64, name='X_indices', shape=None)
    X_data = tf.placeholder(tf.float32, name='X_data', shape=None)
    X_shape = tf.placeholder(tf.int64, name='X_shape', shape=None)

    '''
    Y_indices = tf.placeholder(tf.int64, name='Y_indices', shape=None)
    Y_data = tf.placeholder(tf.float32, name='Y_data', shape=None)
    Y_shape = tf.placeholder(tf.int64, name='Y_shape', shape=None)
    '''

    L_indices = tf.placeholder(tf.int64, name='L_indices', shape=None)
    L_data = tf.placeholder(tf.float32, name='L_data', shape=None)
    L_shape = tf.placeholder(tf.int64, name='L_shape', shape=None)

    X = tf.SparseTensor(indices=X_indices, values=X_data, dense_shape=X_shape)
    Y = tf.placeholder(tf.float32, shape=[None, label_dim])
    laps = tf.SparseTensor(indices=L_indices, values=L_data, dense_shape=L_shape)

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
    hx1 = act(dot(X, Wx1, True) + bx1)
    hxe = act(dot(hx1, Wx2) + bx2)
    
    hy1 = act(dot(Y, Wy1) + by1)
    hye = act(dot(hy1, Wy2) + by2)
   
    hhx1 = act(dot(hxe, Wh1) + bh1)
    hhx2 = dot(hhx1, Wh2) + bh2

    hhy1 = act(dot(hye, Wh1) + bh1)
    hhy2 = dot(hhy1, Wh2) + bh2

    loss1 = ce_loss(hhx2, Y) + ce_loss(hhy2, Y)
    loss2 = dot(tf.transpose(dot(tf.sparse_transpose(laps), hxe, True)), hxe)
    loss3 = dot(tf.transpose(dot(tf.sparse_transpose(laps), hye, True)), hye)

    tf.summary.scalar("loss1", loss1)
    tf.summary.scalar("loss2", loss2)
    tf.summary.scalar("loss3", loss3)

    loss = loss1 + loss2 + loss3
    tf.summary.scalar("loss", loss)
    train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    pat3 = tf.metrics.sparse_precision_at_k(labels=tf.cast(Y, tf.int64), predictions=tf.nn.sigmoid(hhx2), k=3)


    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter("./tensorboard", sess.graph)
        for epoch in range(no_epoch):
            el, c = 0.0, 0
            dataobj = DataSet("./data/delicious/delicious-train", batch_size)
            for x_train, y_train, dummy in dataobj.next_batch("train", sparse_features=True, sparse_labels=True):
                print(y_train.shape)
                laplacian = get_lap(y_train)
                y_train = sp.csr_matrix.todense(y_train)
                x_props, l_props = get_sparse_props(x_train), get_sparse_props(laplacian)
                feed = {X_indices : x_props[0], X_data : x_props[1], X_shape : x_props[2], Y : y_train, L_indices : l_props[0], L_data : l_props[1], L_shape : l_props[2]}
                pl, _, summ = sess.run([loss, train, merged], feed_dict=feed)
                el += pl
                c += 1
                print("Epoch #{} Loss : {}".format(epoch, pl), end='\r')
            saver.save(sess, './checkpoint/model.ckpt', global_step=c)
            writer.add_summary(summ, c)
            test_obj = DataSet("./data/delicious/delicious-test", 3185)
            x_test, y_test = test_obj.get_test()
            x_props, y_props = get_sparse_props(x_test), None #get_sparse_props(y_test)
            feed = {X_indices : x_props[0], X_data : x_props[1], X_shape : x_props[2], Y : y_test} #Y_indices : y_props[0], Y_data : y_props[1], Y_shape : y_props[2]}
            pk = sess.run(patk, feed_dict=feed)
            output = "Epoch #{} Loss : {}, P@K : {}".format(epoch, el/c, pk)
            with open("train_test.log", "a+") as f:
                f.write(output)
            print(output)

if __name__ == '__main__':
    model()