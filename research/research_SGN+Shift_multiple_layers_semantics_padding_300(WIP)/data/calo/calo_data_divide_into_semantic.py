import numpy as np
import pandas as pd
import os.path as osp
import pickle as pkl

key_train = 'train'
key_test = 'test'
path = osp.join('./')

# H5 format
# train_X = pd.read_hdf(path + "data_training_test.h5",
#                            key=key_train + "_data").to_numpy()  # 35763x300x150
# train_Y = pd.read_hdf(path + "data_training_test.h5",
#                            key=key_train + "_label").to_numpy()  # 35763x300x150  # 35763
#
# test_X = pd.read_hdf(path + "data_training_test.h5",
#                           key=key_test + "_data").to_numpy()  # 35763x300x150
# test_Y = pd.read_hdf(path + "data_training_test.h5",
#                           key=key_test + "_label").to_numpy()  # 35763x300x150

# Numpy format
train_X = np.load('train_data_joint.npy')
with open('train_label.pkl', 'rb') as f:
    train_Y = pkl.load(f)
val_X = np.load('val_data_joint.npy')
with open('val_label.pkl', 'rb') as f:
    val_Y = pkl.load(f)

# Upper + Middle part

index_upper_middle = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 14, 15, 16, 17]
train_data_upper_middle = np.stack([train_X[:, :, :, i, :] for i in index_upper_middle]).transpose(1, 2, 3, 0, 4)

val_data_upper_middle = np.stack([val_X[:, :, :, i, :] for i in index_upper_middle]).transpose(1, 2, 3, 0, 4)

with open('train_data_joint_upper_middle.npy', 'wb') as f:
    np.save(f, train_data_upper_middle)

with open('val_data_joint_upper_middle.npy', 'wb') as f:
    np.save(f, val_data_upper_middle)

# Lower + Middle part

index_lower_middle = [1, 2, 5, 8, 9, 10, 11, 12, 13]
train_data_lower_middle = np.stack([train_X[:, :, :, i, :] for i in index_lower_middle]).transpose(1, 2, 3, 0, 4)

val_data_lower_middle = np.stack([val_X[:, :, :, i, :] for i in index_lower_middle]).transpose(1, 2, 3, 0, 4)

with open('train_data_joint_lower_middle.npy', 'wb') as f:
    np.save(f, train_data_lower_middle)

with open('val_data_joint_lower_middle.npy', 'wb') as f:
    np.save(f, val_data_lower_middle)


print("")