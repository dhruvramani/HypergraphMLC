import tensorflow as tf
from src.utils.util import *
from src.utils.inits import *
from src.layers.layer import Layer

class ConvText(Layer):
    def __init__(self, config, input_shape, sparse_inputs=True,
        act=tf.nn.relu, bias=False, **kwargs):
        super(ConvText, self).__init__(**kwargs)

        self.config = config
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.input_shape = input_shape
        self.bias = bias

        with tf.variable_scope(self.namr + '_vars'):
            for i in self.config.filter_sizes:
                filter_shape = [i, self.input_shape[1], 1, self.config.num_filters]
                self.vars['weights_{}'.format(i)] = glorot(filter_shape, name='weights_{}'.format(i))
                if(self.bias):
                    self.vars['bias_{}'.format(i)] = glorot([self.config.num_filters], name='bias_{}'.format(i))

        if self.logging:
            self._log_vars()


    def _call(self, inputs):
        x, outs = inputs['activations'][-1], list()
        for i in self.config.filter_sizes:
            conv = tf.nn.conv2d(x, self.vars['weights_{}'.format(i)], stides=[1,1,1,1], padding='VALID')
            if(self.bias):
                conv = tf.nn.bias_add(conv, self.vars['bias_{}'.format(i)])
            h = tf.nn.relu(conv)
            pooled = tf.nn.max_pool(h, ksize=[1, self.config.sequence_length - i + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
            outs.append(pooled)
        
