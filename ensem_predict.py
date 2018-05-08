import os
import sys
sys.path.append("..")
sys.path.append("../..")

import tensorflow as tf
import numpy as np
from gen_train_val import PulsarDataGenerator

def predict(pre_x1, pre_x2, pre_x3, pre_x4):
    ckpt = tf.train.get_checkpoint_state('tmp/checkpoints/')
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
        new_saver.restore(sess, ckpt.model_checkpoint_path)
        tf.get_default_graph()
        x_1 = sess.graph.get_tensor_by_name('input1:0')
        x_2 = sess.graph.get_tensor_by_name('input2:0')
        x_3 = sess.graph.get_tensor_by_name('input3:0')
        x_4 = sess.graph.get_tensor_by_name('input4:0')
        rate = sess.graph.get_tensor_by_name('rate:0')
        training = sess.graph.get_tensor_by_name('is_training:0')
        y = sess.graph.get_tensor_by_name('output:0')
        result = sess.run(y, feed_dict={x_1: pre_x1, x_2: pre_x2, x_3: pre_x3, x_4: pre_x4, rate: 0., training: False})
        label = np.argmax(result, 1)
        return label

if __name__ == '__main__':
    strFvPs = "../datasets/pfd_data/FvPs_2.pkl"
    strTvPs = "../datasets/pfd_data/TvPs_2.pkl"
    strDMs = "../datasets/pfd_data/DMcs_2.pkl"
    strProfi = "../datasets/pfd_data/profiles_2.pkl"
    strLabels = "../datasets/pfd_data/PulsarFlag_2.pkl"
    pd = PulsarDataGenerator(strFvPs, strTvPs, strDMs, strProfi, strLabels, shuffle=True)
    train_FvPs, val_FvPs, train_TvPs, val_TvPs, train_DMs, val_DMs, train_prof, val_Prof, \
    train_labels, val_labels = pd.gen_train_val_data()
    result = predict(val_FvPs, val_TvPs, val_DMs, val_Prof)
    pred_count = np.equal(val_labels, result)
    precision = np.mean(pred_count)
    print(precision)