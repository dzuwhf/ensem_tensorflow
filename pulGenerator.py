import numpy as np
import pickle
import os

class ImageDataGenerator:
    def __init__(self, fvp_data, tvp_data, dm_data, profi_data, labels,
                 shuffle=False, scale_size=(64, 64), nb_classes=2):
        self.fvp_data = fvp_data
        self.tvp_data = tvp_data
        self.dm_data = dm_data
        self.scale_size = scale_size
        self.profi_data = profi_data
        self.labels = labels
        self.shuffle = shuffle
        self.pointer = 0
        self.nb_classes = nb_classes
        self.data_size = len(labels)

        if self.shuffle:
            self.shuffle_data()

    def shuffle_data(self):
        fvp_data = self.fvp_data
        tvp_data = self.tvp_data
        dm_data = self.dm_data
        profi_data = self.profi_data
        labels = self.labels
        self.fvp_data = []
        self.tvp_data = []
        self.dm_data = []
        self.profi_data = []
        self.labels = []

        idx = np.random.permutation(len(labels))
        for i in idx:
            self.fvp_data.append(fvp_data[i])
            self.tvp_data.append(tvp_data[i])
            self.dm_data.append(dm_data[i])
            self.profi_data.append(profi_data[i])
            self.labels.append(labels[i])

    def reset_pointer(self):
        self.pointer = 0
        if self.shuffle:
            self.shuffle_data()

    def next_batch(self, batch_size):

        path_0 = self.fvp_data[self.pointer:self.pointer + batch_size]
        path_1 = self.tvp_data[self.pointer:self.pointer + batch_size]
        path_2 = self.dm_data[self.pointer:self.pointer + batch_size]
        path_3 = self.profi_data[self.pointer:self.pointer + batch_size]
        labels = self.labels[self.pointer:self.pointer + batch_size]

        self.pointer += batch_size

        images_0 = np.ndarray([batch_size, self.scale_size[0], self.scale_size[1], 1])
        images_1 = np.ndarray([batch_size, self.scale_size[0], self.scale_size[1], 1])
        images_2 = np.ndarray([batch_size, self.scale_size[0], 1])
        images_3 = np.ndarray([batch_size, self.scale_size[0], 1])


        for i in range(len(path_0)):
            img_0 = path_0[i]
            img_0 = img_0.astype(np.float32)
            images_0[i] = img_0
            img_1 = path_1[i]
            img_1 = img_1.astype(np.float32)
            images_1[i] = img_1
            img_2 = path_2[i]
            img_2 = img_2.astype(np.float32)
            images_2[i] = img_2
            img_3 = path_3[i]
            img_3 = img_3.astype(np.float32)
            images_3[i] = img_3

        one_hot_labels = np.zeros((batch_size, self.nb_classes))
        for i in range(len(labels)):
            one_hot_labels[i][labels[i]] = 1

        return images_0, images_1, images_2, images_3, one_hot_labels



