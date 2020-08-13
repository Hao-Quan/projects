import numpy as np
import pandas as pd
import os.path as osp
import pickle as pkl

path = osp.join('./xsub')

# Numpy format
train_X = np.load('./ntu/xsub/train_data_joint.npy', mmap_mode='r')
with open('./ntu/xsub/train_label.pkl', 'rb') as f:
    train_Y = pkl.load(f)
val_X = np.load('./ntu/xsub/val_data_joint.npy', mmap_mode='r')
with open('./ntu/xsub/val_label.pkl', 'rb') as f:
    val_Y = pkl.load(f)

train_data_joint = train_X[0:100]
train_data_label = [train_Y[0][0:100], train_Y[1][0:100]]

val_data_joint = val_X[0:20]
val_data_label = [val_Y[0][0:20], val_Y[1][0:20]]

with open('./ntu_light_25/xsub/train_data_joint.npy', 'wb') as f:
    np.save(f, train_data_joint)

with open('./ntu_light_25/xsub/val_data_joint.npy', 'wb') as f:
    np.save(f, val_data_joint)

with open('./ntu_light_25/xsub/train_label.pkl', 'wb') as f:
    pkl.dump(train_data_label, f)

with open('./ntu_light_25/xsub/val_label.pkl', 'wb') as f:
    pkl.dump(val_data_label, f)

print("")