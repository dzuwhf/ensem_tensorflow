import os
import numpy as np
from ubc_AI.data import pfdreader
import pickle


class GeneratePickle(object):
    def __init__(self, save_path, read_path):
        self.save_path = save_path
        self.read_path = read_path


    def read_filename(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.read_path):
            print('Reading file path is wrong, please set right reading path')

        pfd_name_list = []
        labels = []
        for _, dirs, _ in os.walk(self.read_path):
            for dir in dirs:
                for _, _, files in os.walk(os.path.join(self.read_path, dir)):
                    for file in files:
                        file_path = os.path.join(self.read_path, dir, file)
                        pfd_name_list.append(file_path)
                        labels.append(dir)

        shuffle_pfdlist = []
        shuffle_labels = []
        idx = np.random.permutation(len(labels))
        for i in idx:
            shuffle_pfdlist.append(pfd_name_list[i])
            shuffle_labels.append(labels[i])

        self.pfds_list = shuffle_pfdlist
        self.labels = shuffle_labels
        self.datasize = len(labels)

    def save_pickle(self, file, filesavepath):
        with open(filesavepath, 'wb') as filepath:
            pickle.dump(file, filepath, protocol=2)
            filepath.close()
        print('save {} success !'.format(filesavepath))

    def read_pfd(self):
        pfds_list = self.pfds_list
        labels = self.labels
        count = self.datasize
        profiles = np.empty((count, 1, 64))
        DMcs = np.empty((count, 1, 64))
        TvPs = np.empty((count, 64, 64))
        FvPs = np.empty((count, 64, 64))
        n = 0
        for pfd_file in pfds_list:
            apfd = pfdreader(pfd_file)
            TvP = apfd.getdata(intervals=64).reshape(64, 64)
            FvP = apfd.getdata(subbands=64).reshape(64, 64)
            profile = apfd.getdata(phasebins=64)
            DMc = apfd.getdata(DMbins=64)
            TvPs[n] = TvP
            FvPs[n] = FvP
            profiles[n] = profile
            DMcs[n] = DMc
            n = n + 1
        pulsar_flag = np.array(labels)
        print(pulsar_flag)

        self.save_pickle(TvPs, self.save_path + 'TvPs_2.pkl')
        self.save_pickle(FvPs, self.save_path + 'FvPs_2.pkl')
        self.save_pickle(pulsar_flag, self.save_path + 'PulsarFlag_2.pkl')
        self.save_pickle(profiles, self.save_path + 'profiles_2.pkl')
        self.save_pickle(DMcs, self.save_path + 'DMcs_2.pkl')

if __name__ == '__main__':
    read_path = "../datasets/pfd_orig/"
    save_path = "../datasets/tfrecord/"
    gen_pfd = GeneratePickle(save_path, read_path)
    gen_pfd.read_filename()
    gen_pfd.read_pfd()