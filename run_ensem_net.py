import os
import sys

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
sys.path.append("..")
sys.path.append("../..")

from datetime import datetime
import numpy as np
import time
import tensorflow as tf
import ensem_net
from utils import mkdirs
from pulGenerator import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from gen_train_val import PulsarDataGenerator

class ensem_model(object):

    def __init__(self, image_size, num_epoch, batch_size, learning_rate,
                 weight_decay, num_classes, dropout_rate, filewriter_path, checkpoint_path,
                 is_restore=True):
        self.image_size =image_size
        self.num_epochs = num_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.display = 20
        self.dropout_rate = dropout_rate
        self.filewriter_path = filewriter_path
        mkdirs(self.filewriter_path)
        self.checkpoint_path = checkpoint_path
        mkdirs(self.checkpoint_path)
        if is_restore:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
            self.restore_checkpoint = ckpt.model_checkpoint_path
        else:
            self.restore_checkpoint = ''

    def fit(self, fvp_train, tvp_train, dm_train, prof_train, label_train, fvp_val,
            tvp_val, dm_val, profi_val, label_val):
        x_1 = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 1], name='input1')
        x_2 = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 1], name='input2')
        x_3 = tf.placeholder(tf.float32, [None, self.image_size, 1], name='input3')
        x_4 = tf.placeholder(tf.float32, [None, self.image_size, 1], name='input4')
        y = tf.placeholder(tf.float32, [None, self.num_classes])
        rate = tf.placeholder(tf.float32, name='rate')
        is_training = tf.placeholder(tf.bool, [], name='is_training')
        model_fvp = ensem_net.construct_fvp_2d(x_1, is_training)
        model_tvp = ensem_net.construct_tvp_2d(x_2, is_training)
        model_dm = ensem_net.construct_dm_1d(x_3, is_training)
        model_prof = ensem_net.construct_prof_1d(x_4, is_training)
        model_merge = ensem_net.merge_model(model_fvp, model_tvp, model_dm, model_prof, rate)
        predict = model_merge
        output = tf.nn.softmax(predict, name='output')

        with tf.name_scope("corss_ent"):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))

        # var_list = [v for v in tf.trainable_variables()]
        # with tf.name_scope("train"):
        #     gradients = tf.gradients(cost, var_list)
        #     gradients = list(zip(gradients, var_list))
        #     optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
        #     train_op = optimizer.apply_gradients(grads_and_vars=gradients)
        #
        # for gradient, var in gradients:
        #     tf.summary.histogram(var.name + '/gradient', gradient)
        #
        # for var in var_list:
        #     tf.summary.histogram(var.name, var)
        # for batch normalization as below
        optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(cost)
        # for batch normalization as above
        tf.summary.scalar('corss_entropy', cost)
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(self.filewriter_path)

        with tf.name_scope("accuracy"):
            prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

        saver = tf.train.Saver()
        train_generator = ImageDataGenerator(fvp_train, tvp_train, dm_train, prof_train, label_train,
                                             shuffle=True, scale_size=(self.image_size, self.image_size),
                                             nb_classes=self.num_classes)
        val_generator = ImageDataGenerator(fvp_val, tvp_val, dm_val, profi_val, label_val,
                                           shuffle=True, scale_size=(self.image_size, self.image_size),
                                           nb_classes=self.num_classes)
        train_batches_per_epoch = np.floor(train_generator.data_size / self.batch_size).astype(np.int16)
        val_batches_per_epoch = np.floor(val_generator.data_size / self.batch_size).astype(np.int16)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            writer.add_graph(sess.graph)

            if not self.restore_checkpoint == '':
                saver.restore(sess, self.restore_checkpoint)

            print("{} Start training...".format(datetime.now()))
            print("{} open tensorboard: tensorboard --logdir {} --host localhost --port 6006".
                  format(datetime.now(), self.filewriter_path))

            for epoch in range(self.num_epochs):
                print("Epoch number:{}/{}".format(epoch + 1, self.num_epochs))
                step = 1

                while step < train_batches_per_epoch:
                    batch_fvp, batch_tvp, batch_dm, batch_profi, batch_ys = train_generator.next_batch(self.batch_size)
                    feed_dict = {x_1: batch_fvp, x_2: batch_tvp, x_3: batch_dm, x_4: batch_profi, y: batch_ys,
                                 rate: self.dropout_rate, is_training: True}
                    sess.run(train_op, feed_dict=feed_dict)

                    if step % self.display == 0:
                        loss, acc, s = sess.run([cost, accuracy, merged_summary], feed_dict=feed_dict)
                        writer.add_summary(s, epoch * train_batches_per_epoch + step)
                        print("Iter {}/{}, training mini-batch loss = {:.5f}, training accuracy = {:.5f}".format(
                            step * self.batch_size, train_batches_per_epoch * self.batch_size, loss, acc))
                    step += 1

                # Validate the model on the entire validation set
                print("{} Start validation".format(datetime.now()))
                v_loss = 0.
                v_acc = 0.
                count = 0
                t1 = time.time()

                print("validate batchs {}".format(val_batches_per_epoch))
                for i in range(val_batches_per_epoch):
                    batch_fvp_val, batch_tvp_val, batch_dm_val, batch_profi_val, batch_validy = \
                        val_generator.next_batch(self.batch_size)
                    valid_loss, valid_acc, valid_out = sess.run([cost, accuracy, output],
                                                                feed_dict={x_1: batch_fvp_val, x_2: batch_tvp_val,
                                                                           x_3: batch_dm_val, x_4: batch_profi_val,
                                                                y: batch_validy, rate: 0., is_training: False})
                    v_loss += valid_loss
                    v_acc += valid_acc
                    count += 1

                    y_true = np.argmax(batch_validy, 1)
                    y_pre = np.argmax(valid_out, 1)

                    if i == 0:
                        conf_matrix = confusion_matrix(y_true, y_pre)
                    else:
                        conf_matrix += confusion_matrix(y_true, y_pre)

                v_loss /= count
                v_acc /= count
                t2 = time.time() - t1
                print("Validation loss = {:.4f}, acc = {:.4f}".format(v_loss, v_acc))
                print("Test image {:.4f}ms per image".format(t2 * 1000 / (val_batches_per_epoch * self.batch_size)))
                print(conf_matrix)

                val_generator.reset_pointer()
                train_generator.reset_pointer()

                print("{} Saving checkpoint of model...".format(datetime.now()))
                checkpoint_name = os.path.join(self.checkpoint_path, 'epoch_' + str(epoch))
                saver.save(sess, checkpoint_name)

                print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))


if __name__ =="__main__":
    strFvPs = "../datasets/pfd_data/FvPs_2.pkl"
    strTvPs = "../datasets/pfd_data/TvPs_2.pkl"
    strDMs = "../datasets/pfd_data/DMcs_2.pkl"
    strProfi = "../datasets/pfd_data/profiles_2.pkl"
    strLabels = "../datasets/pfd_data/PulsarFlag_2.pkl"
    pd = PulsarDataGenerator(strFvPs, strTvPs, strDMs, strProfi, strLabels, shuffle=True)
    train_FvPs, val_FvPs, train_TvPs, val_TvPs, train_DMs, val_DMs, train_prof, val_Prof, \
    train_labels, val_labels = pd.gen_train_val_data()

    ensem_mode = ensem_model(
        image_size=64,
        num_epoch=100,
        batch_size=16,
        learning_rate=0.0001,
        weight_decay=0.00002,
        num_classes=2,
        dropout_rate=0.4,
        filewriter_path="tmp/tensorboard",
        checkpoint_path="tmp/checkpoints",
        is_restore=False
    )
    ensem_mode.fit(train_FvPs, train_TvPs, train_DMs, train_prof, train_labels,
                   val_FvPs, val_TvPs, val_DMs, val_Prof, val_labels)