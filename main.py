import os
import numpy as np
import tensorflow as tf

def hypergraph(np_labels):
    incidence = np_labels.T
    weightDiag = np.eye(np_labels.shape[0])
    edgesDiag = np.zeros((np_labels.shape[0], np_labels.shape[0]))
    vertexDiag = np.zeros((np_labels.shape[1], np_labels.shape[1]))
    for i in range(np_labels.shape[1]):
        vertexDiag[i, i] = np.sum(np_labels[:, i])
    for i in range(np_labels.shape[0]):
        edgesDiag[i, i] = np.sum(np_labels[i, :])
    return [incidence, weightDiag, edgesDiag, vertexDiag]

class Model():
    def __init__(self, sequence_length, feature_dim, label_dim, hypergraph):
        self.sequence_length = sequence_length # shape[0] of filters,
        self.feature_dim = feature_dim
        self.embedding_dim = 128 * 3
        self.label_dim = label_dim
        self.hypergraph = hypergraph # List of Numpy Arrays
        self.filter1, self.b1, self.filter2, self.b2, self.filter3, self.b3, self.Wl, self.bl = self.init_weights()

    def init_weights(self):
        filter1 = tf.Variable(tf.random_normal(shape=[3, self.feature_dim, 1, 128]))
        b1 = tf.Variable(tf.random_normal(shape=[128]))
        filter2 = tf.Variable(tf.random_normal(shape=[4, self.feature_dim, 1, 128]))
        b2 = tf.Variable(tf.random_normal(shape=[128]))
        filter3 = tf.Variable(tf.random_normal(shape=[5, self.feature_dim, 1, 128]))
        b3 = tf.Variable(tf.random_normal(shape=[128]))
        Wl = tf.Variable(tf.random_normal(shape=[self.embedding_dim, self.label_dim]))
        bl = tf.Variable(tf.random_normal(shape=[self.label_dim]))
        return filter1, b1, filter2, b2, filter3, b3, Wl, bl

    def forward_pass(self, features):
        filters_size, num_filters = [3, 4, 5], 128
        vectors = list()
        for i in filters_size:
            fil = tf.Variable(tf.random_normal(shape=[i, self.feature_dim, 1, num_filters]))
            bias = tf.Variable(tf.random_normal(shape=[num_filters]))
            conv = tf.nn.conv2d(features, fil, strides=[1, 1, 1, 1], padding='VALID')
            conv = tf.nn.relu(tf.nn.bias_add(conv, bias))
            pool = tf.nn.maxpool(conv, [1, self.sequence_length - i, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
            vectors.append(pool)
        
        num_filters_total = num_filters * len(filters_size)
        embedding = tf.concat(3, pooled_outputs)
        embedding = tf.reshape(embedding, [-1, num_filters_total])
        return embedding, logits

    def laplacian(self):
        # 𝑳 = 𝑰 - 𝑫ᵥ^(-1/2).𝑯.𝑾.𝑫ℯ⁻¹.𝑯ᵀ.𝑫ᵥ^(-1/2) 
        H, W, De, Dv = self.hypergraph
        L = tf.constant(tf.eye(labels.get_shape().as_list()[0])) - tf.matmul(tf.sqrt(tf.reciprocal(Dv)), tf.matmul(H, tf.matmul(tf.matrix_inverse(De), tf.matmul(tf.transpose(H), tf.sqrt(tf.reciprocal(Dv))))))
        return L

    def loss(self, logits, labels, embeddings, hyp_const):
        lap = self.laplacian()
        cross_loss = tf.add(tf.log(1e-10 + tf.nn.sigmoid(logits)) * labels, tf.log(1e-10 + (1 - tf.nn.sigmoid(logits))) * (1 - labels))
        lap_loss = 2 * tf.trace(tf.matmul(tf.transpose(embeddings), tf.matmul(lap, embeddings)))
        return cross_loss + hyp_const * lap_loss

    def optimize(self, loss, learning_rate):
        optim = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return optim