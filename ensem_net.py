from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

import sys
import numpy as np

#Construct 2D frequency vs phase, then flatten one dimesion 4*4*64

def construct_fvp_2d(X_train, is_training, dp_rate=0.0):
    with tf.variable_scope('con_fvp'):
        conv1 = tf.layers.conv2d(X_train, 16, 5, strides=(2, 2), padding='same', activation=tf.nn.leaky_relu, name='conv1')
        conv1_bn = tf.layers.batch_normalization(conv1, training=is_training, name='conv1_bn')
        conv1_pl = tf.layers.max_pooling2d(conv1_bn, 2, 2, name='pool1')
        conv1_dp = tf.layers.dropout(conv1_pl, rate=dp_rate, training=is_training, name='conv1_dp')
        conv2 = tf.layers.conv2d(conv1_dp, 32, 5, padding='same', activation=tf.nn.leaky_relu, name='conv2')
        conv2_bn = tf.layers.batch_normalization(conv2, training=is_training, name='conv2_bn')
        conv2_pl = tf.layers.max_pooling2d(conv2_bn, 2, 2, name='pool2')
        conv2_dp = tf.layers.dropout(conv2_pl, rate=dp_rate, training=is_training, name='conv2_dp')
        conv3 = tf.layers.conv2d(conv2_dp, 64, 5, padding='same', activation=tf.nn.leaky_relu, name='conv3')
        conv3_bn = tf.layers.batch_normalization(conv3, training=is_training, name='conv3_bn')
        conv3_pl = tf.layers.max_pooling2d(conv3_bn, 2, 2, name='pool3')
        conv3_dp = tf.layers.dropout(conv3_pl, rate=dp_rate, training=is_training, name='conv3_dp')
        out = tf.contrib.layers.flatten(conv3_dp)

    return out
#Construct 2D time vs phase, then flatten one dimesion 4*4*64
def construct_tvp_2d(X_trian, is_training, dp_rate=0.0):
    with tf.variable_scope('con_tvp'):
        conv1 = tf.layers.conv2d(X_trian, 16, 5, strides=(2, 2), padding='same', activation=tf.nn.leaky_relu, name='conv1')
        conv1_bn = tf.layers.batch_normalization(conv1, training=is_training, name='conv1_bn')
        conv1_pl = tf.layers.max_pooling2d(conv1_bn, 2, 2, name='poo11')
        conv1_dp = tf.layers.dropout(conv1_pl, rate=dp_rate, training=is_training, name='conv1_dp')
        conv2 = tf.layers.conv2d(conv1_dp, 32, 5, padding='same', activation=tf.nn.leaky_relu, name='conv2')
        conv2_bn = tf.layers.batch_normalization(conv2, training=is_training, name='conv2_bn')
        conv2_pl = tf.layers.max_pooling2d(conv2_bn, 2, 2, name='pool2')
        conv2_dp = tf.layers.dropout(conv2_pl, rate=dp_rate, training=is_training, name='conv2_dp')
        conv3 = tf.layers.conv2d(conv2_dp, 64, 5, padding='same', activation=tf.nn.leaky_relu, name='conv3')
        conv3_bn = tf.layers.batch_normalization(conv3, training=is_training, name='conv3_bn')
        conv3_pl = tf.layers.max_pooling2d(conv3_bn, 2, 2, name='pool3')
        conv3_dp = tf.layers.dropout(conv3_pl, rate=dp_rate, training=is_training, name='conv3_dp')
        out = tf.contrib.layers.flatten(conv3_dp)

    return out

#Construct 1D DM 16*64
def construct_dm_1d(X_train, is_training, dp_rate=0.0):
    with tf.variable_scope('con_dm'):
        conv1 = tf.layers.conv1d(X_train, 32, 3, padding='same', activation=tf.nn.leaky_relu, name='conv1')
        conv1_bn = tf.layers.batch_normalization(conv1, training=is_training, name='conv1_bn')
        conv1_pl = tf.layers.max_pooling1d(conv1_bn, 2, 2, name='pool1')
        conv1_dp = tf.layers.dropout(conv1_pl, rate=dp_rate, training=is_training, name='conv1_dp')
        conv2 = tf.layers.conv1d(conv1_dp, 64, 3, padding='same', activation=tf.nn.leaky_relu, name='conv2')
        conv2_bn = tf.layers.batch_normalization(conv2, training=is_training, name='conv2_bn')
        conv2_pl = tf.layers.max_pooling1d(conv2_bn, 2, 2, name='pool2')
        conv2_dp = tf.layers.dropout(conv2_pl, rate=dp_rate, training=is_training, name='conv2_dp')
        out = tf.contrib.layers.flatten(conv2_dp)

    return out

#Construct 1D profiles 16*64
def construct_prof_1d(X_train, is_training, dp_rate=0.0):
    with tf.variable_scope('con_profi'):
        conv1 = tf.layers.conv1d(X_train, 32, 3, padding='same', activation=tf.nn.leaky_relu, name='conv1')
        conv1_bn = tf.layers.batch_normalization(conv1, training=is_training, name='conv1_bn')
        conv1_pl = tf.layers.max_pooling1d(conv1_bn, 2, 2, name='pool1')
        conv1_dp = tf.layers.dropout(conv1_pl, rate=dp_rate, training=is_training)
        conv2 = tf.layers.conv1d(conv1_dp, 64, 3, padding='same', activation=tf.nn.leaky_relu, name='conv2')
        conv2_bn = tf.layers.batch_normalization(conv2, training=is_training, name='conv2_bn')
        conv2_pl = tf.layers.max_pooling1d(conv2_bn, 2, 2, name='pool2')
        conv2_dp = tf.layers.dropout(conv2_pl, rate=dp_rate, training=is_training, name='conv2_dp')
        out = tf.contrib.layers.flatten(conv2_dp)

    return out
# Merge model array shape [batch_size, 1024, 1]
def merge_model(model1, model2, model3, model4,is_training, rate=0.):
    with tf.variable_scope('merge_model'):
        model = tf.concat([model1, model2, model3, model4], axis=1)
        # model = tf.add_n([model1, model2, model3, model4])
        # model = tf.squeeze(model, axis=1)
        model_drop_1 = tf.layers.dropout(model, training=is_training, rate=rate)
        model_fc_1 = tf.layers.dense(model_drop_1, 256, activation=tf.nn.relu)
        mode_drop_2 = tf.layers.dropout(model_fc_1, training=is_training, rate=rate)
        logits = tf.layers.dense(mode_drop_2, 2, activation=tf.nn.relu)

    return logits



