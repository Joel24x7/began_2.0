import tensorflow as tf
import numpy as np
import os

#Encoder/Decoder Layers
def conv_layer(input_layer, layer_depth, kernel_size=(3,3), stride=(1,1), stddev=0.02, in_dim=None, padding='SAME', scope='conv_layer'):
    with tf.variable_scope(scope):
        filter_depth = in_dim or input_layer.shape[-1]
        weights = tf.get_variable('weights', 
            [kernel_size[0], kernel_size[1], filter_depth, layer_depth], 
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        bias = tf.get_variable('bias', 
            shape=layer_depth, 
            initializer=tf.constant_initializer(0))
        conv = tf.nn.conv2d(input_layer, 
            weights, 
            strides=[1,stride[0], stride[1], 1], 
            padding=padding)
        conv = tf.nn.bias_add(conv, bias)
        return conv

def dense_layer(input_layer, units, scope='dense', in_dim = None, stddev=0.02, bias_start=0):
    shape = input_layer.shape
    if len(shape) > 2:
        input_layer = tf.reshape(input_layer, [-1, int(np.prod(shape[1:]))])
    shape = input_layer.shape 
    
    with tf.variable_scope(scope):
        num_input_entries = in_dim or shape[1]
        weight_matrix = tf.get_variable('weight_matrix', 
            [num_input_entries, units],
            dtype=tf.float32, 
            initializer=tf.random_normal_initializer(stddev=stddev))
        bias_vector = tf.get_variable('bias_vector', [units], initializer=tf.constant_initializer(bias_start))
        return tf.nn.bias_add(tf.matmul(input_layer, weight_matrix), bias_vector)

#Upsample using Nearest Neighbors
def upsample(conv, size):
    return tf.image.resize_nearest_neighbor(conv, size)

#Downsample with Strided Convolution
def strided_conv_subsample(conv, filters, scope):
    subsampled = conv_layer(input_layer=conv, layer_depth=filters, scope=scope)
    return tf.nn.relu(subsampled)
    
def subsample(conv):
    return tf.nn.avg_pool(conv, ksize = (1,2,2,1), strides = (1,2,2,1), padding='SAME')

#L1 Pixel-wise Loss for distributions
def l1_loss(original_images, reconstructed_images):
    return tf.reduce_mean(tf.abs(original_images-reconstructed_images))
