import numpy as np

import pickle

class PulsarDataGenerator:
    def __init__(self, strFvPs, strTvPs, strDM, strProfi, strTarget, shuffle=False):
        self.shuffle = shuffle
        self.read_class_list(strFvPs, strTvPs, strDM, strProfi, strTarget)
        if self.shuffle:
            self.shuffle_data()

    def load_pickle(self, picklepath):
        with open(picklepath, "rb") as file:
            data = pickle.load(file)
        return data

    def read_class_list(self, strFvPs, strTvPs, strDM, strProfi, strTarget):
        self.FvPs = []
        self.TvPs = []
        self.DMs = []
        self.Profils = []
        self.labels = []
        arr_FvPs = self.load_pickle(strFvPs)
        new_FvPs = np.array(arr_FvPs)
        arr_TvPs = self.load_pickle(strTvPs)
        new_TvPs = np.array(arr_TvPs)
        arr_DMs = self.load_pickle(strDM)
        new_DMs = np.array(arr_DMs)
        arr_profi = self.load_pickle(strProfi)
        new_profi = np.array(arr_profi)
        target = self.load_pickle(strTarget)
        self.labels = np.array(target)
        for i in range(self.labels.size):
            data_FvP = np.empty([64, 64, 1])
            data_FvP[:, :, 0] = new_FvPs[i]
            self.FvPs.append(data_FvP)
            data_TvP = np.empty([64, 64, 1])
            data_TvP[:, :, 0] = new_TvPs[i]
            self.TvPs.append(data_TvP)
            data_DM = np.empty([64, 1])
            data_DM[:, :] = np.transpose(new_DMs[i])
            self.DMs.append(data_DM)
            data_profi = np.empty([64, 1])
            data_profi[:, :] = np.transpose(new_profi[i])
            self.Profils.append(data_profi)
        self.data_size = len(self.labels)

    def shuffle_data(self):
        FvPs = self.FvPs
        TvPs = self.TvPs
        DMs = self.DMs
        Profi = self.Profils
        labels = self.labels

        self.FvPs = []
        self.TvPs = []
        self.DMs = []
        self.Profils = []
        self.labels = []

        idx = np.random.permutation(self.data_size)
        for i in idx:
            self.FvPs.append(FvPs[i])
            self.TvPs.append(TvPs[i])
            self.DMs.append(DMs[i])
            self.Profils.append(Profi[i])
            self.labels.append(labels[i])

    def gen_train_val_data(self, train_ratio=0.8):
        train_FvPs = self.FvPs[:np.int(train_ratio * self.data_size)]
        val_FvPs = self.FvPs[np.int(train_ratio * self.data_size):]
        train_TvPs = self.TvPs[:np.int(train_ratio * self.data_size)]
        val_TvPs = self.TvPs[np.int(train_ratio * self.data_size):]
        train_DMs = self.DMs[:np.int(train_ratio * self.data_size)]
        val_DMs = self.DMs[np.int(train_ratio * self.data_size):]
        train_prof = self.Profils[:np.int(train_ratio * self.data_size)]
        val_Prof = self.Profils[np.int(train_ratio * self.data_size):]
        train_labels = self.labels[:np.int(train_ratio * self.data_size)]
        val_labels = self.labels[np.int(train_ratio * self.data_size):]
        return train_FvPs, val_FvPs, train_TvPs, val_TvPs, train_DMs, val_DMs, train_prof, val_Prof, \
               train_labels, val_labels

if __name__ == '__main__':
    strFvPs = "../datasets/pfd_data/FvPs_2.pkl"
    strTvPs = "../datasets/pfd_data/TvPs_2.pkl"
    strDMs = "../datasets/pfd_data/DMcs_2.pkl"
    strProfi = "../datasets/pfd_data/profiles_2.pkl"
    strLabels = "../datasets/pfd_data/PulsarFlag_2.pkl"
    pd = PulsarDataGenerator(strFvPs, strTvPs, strDMs, strProfi, strLabels, shuffle=True)
    train_FvPs, val_FvPs, train_TvPs, val_TvPs, train_DMs, val_DMs, train_prof, val_Prof, \
    train_labels, val_labels = pd.gen_train_val_data()