import numpy as np
import pandas as pd
import os.path as osp
import pickle as pkl

key_train = 'train'
key_test = 'test'
path = osp.join('./')

# H5 format
train_X = pd.read_hdf(path + "data_training_test.h5",
                           key=key_train + "_data").to_numpy()  # 35763x300x150
train_Y = pd.read_hdf(path + "data_training_test.h5",
                           key=key_train + "_label").to_numpy()  # 35763x300x150  # 35763

test_X = pd.read_hdf(path + "data_training_test.h5",
                          key=key_test + "_data").to_numpy()  # 35763x300x150
test_Y = pd.read_hdf(path + "data_training_test.h5",
                          key=key_test + "_label").to_numpy()  # 35763x300x150

index_upper_middle = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22, 23, 28, 29, 30, 31, 32,
                              33, 34, 35]
train_x_upper_middle = np.stack([train_X[:, i] for i in index_upper_middle]).transpose(1, 0)

test_x_upper_middle = np.stack([test_X[:, i] for i in index_upper_middle]).transpose(1, 0)

with open('train_data_joint_upper_middle.npy', 'wb') as f:
    np.save(f, train_x_upper_middle)

with open('train_label_upper_middle.pkl', 'wb') as f:
    pkl.dump(train_Y, f)

with open('test_data_joint_upper_middle.npy', 'wb') as f:
    np.save(f, test_x_upper_middle)

with open('test_label_upper_middle.pkl', 'wb') as f:
    pkl.dump(test_Y, f)


print("")