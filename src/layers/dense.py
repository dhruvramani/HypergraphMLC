import tensorflow as tf
from src.utils.util import *
from src.utils.inits import *
from src.layers.layer import Layer

class Dense(Layer):
    def __init__(self, input_dim, output_dim, nnz_features, dropout, sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.dropout = dropout
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.nnz_features = nnz_features
        # self.vars - is defined in super-class
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim], name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs['activations'][-1]
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.nnz_features)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)
        if self.bias:
            output += self.vars['bias']

        h = self.act(output)
        return h