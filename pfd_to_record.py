import os
import numpy as np
from ubc_AI.data import pfdreader
import tensorflow as tf

def write_record(save_floder, read_path):

    if not os.path.exists(save_floder):
        os.makedirs(save_floder)
    if not os.path.exists(read_path):
        print("File path wrong, please set right reading path")

    pfd_list = []
    labels = []
    for _, dirs, _ in os.walk(read_path):
        for dir in dirs:
            for _, _, files in os.walk(os.path.join(read_path, dir)):
                for file in files:
                    file_path = os.path.join(read_path, dir, file)
                    pfd_list.append(file_path)
                    labels.append(dir)
    shuffle_pfdlist = []
    shuffle_labels = []

    idx = np.random.permutation(len(labels))

    for i in idx:
        shuffle_pfdlist.append(pfd_list[i])
        shuffle_labels.append(labels[i])


    train_file_num = np.int(0.8*len(shuffle_labels))
    n = 0
    train_record_path = save_floder + "train_pfd.tfrecords"
    valid_record_path = save_floder + "valid_pfd.tfrecords"
    train_pfd_writer = tf.python_io.TFRecordWriter(train_record_path)
    vaild_pfd_writer = tf.python_io.TFRecordWriter(valid_record_path)

    for pfd_file in shuffle_pfdlist:
        apfd = pfdreader(pfd_file)
        TvP = apfd.getdata(intervals=64).reshape(64, 64)
        new_TvP = np.array(TvP)
        data_TvP = np.empty([64, 64, 1])
        data_TvP[:, :, 0] = new_TvP
        tvp_raw = data_TvP.tostring()
        FvP = apfd.getdata(subbands=64).reshape(64, 64)
        new_FvP = np.array(FvP)
        data_FvP = np.empty([64, 64, 1])
        data_FvP[:, :, 0] = new_FvP
        fvp_raw = data_FvP.tostring()
        profile = apfd.getdata(phasebins=64)
        new_profile = np.array(profile)
        data_profile = np.empty([64, 1])
        data_profile[:, 0] = np.transpose(new_profile)
        prof_raw = data_profile.tostring()
        dmb = apfd.getdata(DMbins=64)
        new_dmb = np.array(dmb)
        data_dmb = np.empty([64, 1])
        data_dmb[:, 0] = np.transpose(new_dmb)
        dm_raw = data_dmb.tostring()
        raw_label = np.int64(shuffle_labels[n])


        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                'fvp': tf.train.Feature(bytes_list=tf.train.BytesList(value=[fvp_raw])),
                'tvp': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tvp_raw])),
                'prof': tf.train.Feature(bytes_list=tf.train.BytesList(value=[prof_raw])),
                'dm': tf.train.Feature(bytes_list=tf.train.BytesList(value=[dm_raw])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[raw_label]))}
            )
        )
        if n < train_file_num:
            train_pfd_writer.write(example.SerializeToString())
        else:
            vaild_pfd_writer.write(example.SerializeToString())
        n += 1

    train_pfd_writer.close()
    vaild_pfd_writer.close()
    print("pfd data to tfrecord data have finished!")


def read_and_decode(filename_queue, is_batch, batch_size):
    # filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'fvp': tf.FixedLenFeature([], tf.string),
                                           'tvp': tf.FixedLenFeature([], tf.string),
                                           'prof': tf.FixedLenFeature([], tf.string),
                                           'dm': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64)
                                       })

    fvp = tf.decode_raw(features['fvp'], tf.float64)
    fvp = tf.reshape(fvp, [64, 64, 1])
    fvp = tf.cast(fvp, tf.float64)

    tvp = tf.decode_raw(features['tvp'], tf.float64)
    tvp = tf.reshape(tvp, [64, 64, 1])
    tvp = tf.cast(tvp, tf.float64)

    prof = tf.decode_raw(features['prof'], tf.float64)
    prof = tf.reshape(prof, [64, 1])
    prof = tf.cast(prof, tf.float64)

    dm = tf.decode_raw(features['dm'], tf.float64)
    dm = tf.reshape(dm, [64, 1])
    dm = tf.cast(dm, tf.float64)


    label = tf.cast(features['label'], tf.int64)

    label = tf.one_hot(label, 2, 1, 0)

    if is_batch:
        batch_size = batch_size
        min_after_dequeue = 10
        capacity = min_after_dequeue + 3*batch_size
        fvp, tvp, prof, dm, label = tf.train.shuffle_batch([fvp, tvp, prof, dm, label],
                                                           batch_size=batch_size,
                                                           num_threads=3,
                                                           capacity=capacity,
                                                           min_after_dequeue=min_after_dequeue)

    return fvp, tvp, prof, dm, label

if __name__ == '__main__':

    # save_path = "../datasets/tfrecord/"
    # read_path = "../datasets/pfd_orig/"
    # write_record(save_path, read_path)
    train_filename = "../datasets/tfrecord/train_pfd.tfrecords"
    test_filename = "../datasets/tfrecord/valid_pfd.tfrecords"
    filename_queue = tf.train.string_input_producer([train_filename], num_epochs=None)
    train_fvps, train_tvps, train_profs, train_dms, train_labels = read_and_decode(filename_queue,
                                                                              is_batch=True, batch_size=16)

    filename_queue = tf.train.string_input_producer([test_filename], num_epochs=None)
    val_fvps, val_tvps, val_profs, val_dms, val_labels = read_and_decode(filename_queue,
                                                                         is_batch=False, batch_size=16)
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            for i in range(2):
                # t_1, t_2, t_3, t_4, t_5 = sess.run([train_fvps, train_tvps, train_profs, train_dms, train_labels])
                # print("train dataset")
                # print(t_5)
                v_1, v_2, v_3, v_4, v_5 = sess.run([val_fvps, val_tvps, val_profs, val_dms, val_labels])
                print(v_5)
        except tf.errors.OutOfRangeError:
            print('done reading')
        finally:
            coord.request_stop()
        coord.request_stop()
        coord.join(threads)