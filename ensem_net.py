from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

import sys
import numpy as np

#Construct 2D time vs phase, then flatten one dimesion None*1*1*2

def construct_fvp_2d(X_train, is_training=True):
    with tf.variable_scope('con_fvp'):
        conv1 = tf.layers.conv2d(X_train, filters=16, kernel_size=5, strides=(2, 2), padding='same',
                                 activation=tf.nn.relu, name='conv1')
        # conv1_bn = tf.layers.batch_normalization(conv1, training=is_training, name='conv1_bn')
        conv1_pl = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2, name='pool1')
        conv2 = tf.layers.conv2d(conv1_pl, filters=32, kernel_size=5, padding='same', activation=tf.nn.relu,
                                 name='conv2')
        # conv2_bn = tf.layers.batch_normalization(conv2, training=is_training, name='conv2_bn')
        conv2_pl = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2, name='pool2')
        conv3 = tf.layers.conv2d(conv2_pl, filters=64, kernel_size=5, strides=(2, 2), padding='same',
                                 activation=tf.nn.relu, name='conv3')
        # conv3_bn = tf.layers.batch_normalization(conv3, training=is_training, name='conv3_bn')
        conv3_pl = tf.layers.max_pooling2d(conv3, pool_size=2, strides=2, name='pool3')
        out = tf.layers.conv2d(conv3_pl, filters=2, kernel_size=2, strides=(2, 2), padding='same',
                               activation=tf.nn.relu, name='conv4')

    return out
#Construct 2D time vs phase, then flatten one dimesion None*1*1*2
def construct_tvp_2d(X_train, is_training=True):
    with tf.variable_scope('con_tvp'):
        conv1 = tf.layers.conv2d(X_train, filters=16, kernel_size=5, strides=(2, 2), padding='same',
                                 activation=tf.nn.relu, name='conv1')
        # conv1_bn = tf.layers.batch_normalization(conv1, training=is_training, name='conv1_bn')
        conv1_pl = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2, name='pool1')
        conv2 = tf.layers.conv2d(conv1_pl, filters=32, kernel_size=5, padding='same', activation=tf.nn.relu,
                                 name='conv2')
        # conv2_bn = tf.layers.batch_normalization(conv2, training=is_training, name='conv2_bn')
        conv2_pl = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2, name='pool2')
        conv3 = tf.layers.conv2d(conv2_pl, filters=64, kernel_size=5, strides=(2, 2), padding='same',
                                 activation=tf.nn.relu, name='conv3')
        # conv3_bn = tf.layers.batch_normalization(conv3, training=is_training, name='conv3_bn')
        conv3_pl = tf.layers.max_pooling2d(conv3, pool_size=2, strides=2, name='pool3')
        out = tf.layers.conv2d(conv3_pl, filters=2, kernel_size=2, strides=(2, 2), padding='same',
                               activation=tf.nn.relu, name='conv4')

    return out

#Construct 1D DM None*1*2
def construct_dm_1d(X_train, is_training=True):
    with tf.variable_scope('con_dm'):
        conv1 = tf.layers.conv1d(X_train, filters=32, kernel_size=3, strides=2, padding='same',
                                 activation=tf.nn.relu, name='conv1')
        # conv1_bn = tf.layers.batch_normalization(conv1, training=is_training, name='conv1_bn')
        conv1_pl = tf.layers.max_pooling1d(conv1, pool_size=2, strides=2, name='pool1')
        conv2 = tf.layers.conv1d(conv1_pl, filters=64, kernel_size=3, strides=2, padding='same',
                                 activation=tf.nn.relu, name='conv2')
        # conv2_bn = tf.layers.batch_normalization(conv2, training=is_training, name='conv2_bn')
        conv2_pl = tf.layers.max_pooling1d(conv2, pool_size=2, strides=2, name='pool2')
        out = tf.layers.conv1d(conv2_pl, filters=2, kernel_size=4, activation=tf.nn.relu, name='conv3')

    return out

#Construct 1D DM None*1*2
def construct_prof_1d(X_train, is_training=True):
    with tf.variable_scope('con_profi'):
        conv1 = tf.layers.conv1d(X_train, filters=32, kernel_size=3, strides=2, padding='same',
                                 activation=tf.nn.relu, name='conv1')
        # conv1_bn = tf.layers.batch_normalization(conv1, training=is_training, name='conv1_bn')
        conv1_pl = tf.layers.max_pooling1d(conv1, pool_size=2, strides=2, name='pool1')
        conv2 = tf.layers.conv1d(conv1_pl, filters=64, kernel_size=3, strides=2, padding='same',
                                 activation=tf.nn.relu, name='conv2')
        # conv2_bn = tf.layers.batch_normalization(conv2, training=is_training, name='conv2_bn')
        conv2_pl = tf.layers.max_pooling1d(conv2, pool_size=2, strides=2, name='pool2')
        out = tf.layers.conv1d(conv2_pl, filters=2, kernel_size=4, activation=tf.nn.relu, name='conv3')

    return out

#output_shape: [None, 2]
def merge(model1, model2, model3, model4):
    model3 = tf.expand_dims(model3, axis=1)
    model4 = tf.expand_dims(model4, axis=1)
    out = tf.concat([model1, model2, model3, model4], axis=3)
    out = tf.layers.conv2d(out, filters=2, kernel_size=1)
    out = tf.squeeze(out, [1, 2])
    return out


# Merge model array shape [batch_size, 1024, 1]
# def merge_model(model1, model2, model3, model4,is_training, rate=0.):
#     with tf.variable_scope('merge_model'):
#         model = tf.concat([model1, model2, model3, model4], axis=1)
#         # model = tf.add_n([model1, model2, model3, model4])
#         # model = tf.squeeze(model, axis=1)
#         model_drop_1 = tf.layers.dropout(model, training=is_training, rate=rate)
#         model_fc_1 = tf.layers.dense(model_drop_1, 256, activation=tf.nn.relu)
#         mode_drop_2 = tf.layers.dropout(model_fc_1, training=is_training, rate=rate)
#         logits = tf.layers.dense(mode_drop_2, 2, activation=tf.nn.relu)
#
#     return logits



if __name__ == '__main__':
    x1 = tf.placeholder(tf.float32, [None, 64, 64, 1])
    x2 = tf.placeholder(tf.float32, [None, 64, 64, 1])
    x3 = tf.placeholder(tf.float32, [None, 64, 1])
    x4 = tf.placeholder(tf.float32, [None, 64, 1])
    out1 = construct_fvp_2d(x1, True)
    out2 = construct_tvp_2d(x2)
    out3 = construct_dm_1d(x3)
    out4 = construct_prof_1d(x4)
    print(out1, out2, out3, out4)
    out = merge(out1, out2, out3, out4)
    print(out)