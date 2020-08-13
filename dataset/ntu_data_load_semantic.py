import numpy as np
import pandas as pd
import os.path as osp
import pickle as pkl

path = osp.join('./xsub')

# Numpy format
train_X = np.load('ntu/xsub/train_data_joint_upper_middle.npy', mmap_mode='r')
with open('ntu/xsub/train_label.pkl', 'rb') as f:
    train_Y = pkl.load(f)
val_X = np.load('ntu/xsub/val_data_joint_upper_middle.npy', mmap_mode='r')
with open('ntu/xsub/val_label.pkl', 'rb') as f:
    val_Y = pkl.load(f)


print("")