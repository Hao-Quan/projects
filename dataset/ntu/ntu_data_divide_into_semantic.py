import numpy as np
import pandas as pd
import os.path as osp
import pickle as pkl

path = osp.join('./xsub')

# Numpy format
train_X = np.load('./xsub/train_data_joint.npy', mmap_mode='r')
with open('./xsub/train_label.pkl', 'rb') as f:
    train_Y = pkl.load(f)
val_X = np.load('./xsub/val_data_joint.npy', mmap_mode='r')
with open('./xsub/val_label.pkl', 'rb') as f:
    val_Y = pkl.load(f)

# Upper + Middle part

index_upper_middle = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20, 21, 22, 23, 24]
train_data_upper_middle = np.stack([train_X[:, :, :, i, :] for i in index_upper_middle]).transpose(1, 2, 3, 0, 4)

val_data_upper_middle = np.stack([val_X[:, :, :, i, :] for i in index_upper_middle]).transpose(1, 2, 3, 0, 4)

with open('./xsub/train_data_joint_upper_middle.npy', 'wb') as f:
    np.save(f, train_data_upper_middle)

with open('./xsub/val_data_joint_upper_middle.npy', 'wb') as f:
    np.save(f, val_data_upper_middle)

# Lower + Middle part

index_lower_middle = [0, 1, 4, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20]
train_data_lower_middle = np.stack([train_X[:, :, :, i, :] for i in index_lower_middle]).transpose(1, 2, 3, 0, 4)

val_data_lower_middle = np.stack([val_X[:, :, :, i, :] for i in index_lower_middle]).transpose(1, 2, 3, 0, 4)

with open('./xview/train_data_joint_lower_middle.npy', 'wb') as f:
    np.save(f, train_data_lower_middle)

with open('./xview/val_data_joint_lower_middle.npy', 'wb') as f:
    np.save(f, val_data_lower_middle)


print("")