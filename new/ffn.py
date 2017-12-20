import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from dataset import DataSet

batch_size, feature_dim, label_dim = 300, 500, 983
def dot(x, y, sparse=True):
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

def model():
    X_indices = tf.placeholder(tf.int64, name='X_indices', shape=None)
    X_data = tf.placeholder(tf.float32, name='X_data', shape=None)
    X_shape = tf.placeholder(tf.int64, name='X_shape', shape=None)

    '''
    Y_indices = tf.placeholder(tf.int64, name='Y_indices', shape=None)
    Y_data = tf.placeholder(tf.float32, name='Y_data', shape=None)
    Y_shape = tf.placeholder(tf.int64, name='Y_shape', shape=None)
    '''

    X = tf.SparseTensor(indices=X_indices, values=X_data, dense_shape=X_shape)
    #Y = tf.SparseTensor(indices=Y_indices, values=Y_data, dense_shape=Y_shape)
    Y = tf.placeholder(tf.float32, shape=[None, label_dim])
    Wx1 = tf.Variable(tf.random_normal(shape=[feature_dim, 700]))
    bx1 = tf.Variable(tf.random_normal(shape=[700]))
    Wx2 = tf.Variable(tf.random_normal(shape=[700, 983]))
    bx2 = tf.Variable(tf.random_normal(shape=[983])) 

    act = tf.nn.relu
    hx1 = act(dot(X, Wx1) + bx1)
    hxe = dot(hx1, Wx2, sparse=False) + bx2
    print(hxe.get_shape())
    loss = ce_loss(hxe, Y)
    patk = tf.metrics.sparse_precision_at_k(labels=tf.cast(Y, tf.int64), predictions=tf.nn.sigmoid(hxe), k=3)

    train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for epoch in range(200):
            el, c = 0.0, 0
            dataobj = DataSet("./data/delicious/delicious-train", batch_size)
            for x_train, y_train, dummy in dataobj.next_batch("train", sparse_features=True, sparse_labels=False):
                x_props, y_props = get_sparse_props(x_train), None #get_sparse_props(y_train)
                feed = {X_indices : x_props[0], X_data : x_props[1], X_shape : x_props[2], Y: y_train} #, Y_indices : y_props[0], Y_data : y_props[1], Y_shape : y_props[2]}
                pl, _ = sess.run([loss, train], feed_dict=feed)
                el += pl
                c += 1
                print("Epoch #{} Loss : {}".format(epoch, pl), end='\r')
            test_obj = DataSet("./data/delicious/delicious-test", 3185)
            x_test, y_test = test_obj.get_test()
            x_props, y_props = get_sparse_props(x_test), None #get_sparse_props(y_test)
            feed = {X_indices : x_props[0], X_data : x_props[1], X_shape : x_props[2], Y : y_test} #Y_indices : y_props[0], Y_data : y_props[1], Y_shape : y_props[2]}
            pk = sess.run(patk, feed_dict=feed)
            print("Epoch #{} Loss : {}, P@K : {}".format(epoch, el/c, pk))

if __name__ == "__main__":
    model()