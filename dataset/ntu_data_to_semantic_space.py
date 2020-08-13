import numpy as np
import pandas as pd
import os.path as osp
import pickle as pkl

'''
Aim to create semantic space data.
it extracts upper+middle and lower+middle space.
- train_data_joint_upper_middle.npy   (40091,3,300,25,2)
- val_data_joint_upper_middle.npy     (16787,3,300,25,2) 

maintain values upper+middle joints: 1,2,3,4,5,6,7,8,9,10,11,12,13,17,21,22,23,24,25
zero all lower values: 14, 15, 16, 18, 19, 20
- train_data_joint_upper_middle.npy   (40097,3,300,25,2)
- val_data_joint_upper_middle.npy     (16487,3,300,25,2) 

'''

path = osp.join('./xsub')

# Numpy format
train_X = np.load('./ntu/xsub/train_data_joint.npy', mmap_mode='r+')
with open('./ntu/xsub/train_label.pkl', 'rb') as f:
    train_Y = pkl.load(f)
val_X = np.load('./ntu/xsub/val_data_joint.npy', mmap_mode='r+')
with open('./ntu/xsub/val_label.pkl', 'rb') as f:
    val_Y = pkl.load(f)

train_data_joint = train_X
val_data_joint = val_X

# joints in upper+middle space
lower_joints = [14, 15, 16, 18, 19, 20]
lower_joints = [k-1 for k in lower_joints]
# Zero all points in lower part, maintain upper+middle part values
for i in lower_joints:
    train_data_joint[:, :, :, i, :] = 0
    val_data_joint[:, :, :, i, :] = 0

with open('./ntu_25/xsub/train_data_joint_upper_middle.npy', 'wb') as f:
    np.save(f, train_data_joint)

with open('./ntu_25/xsub/val_data_joint_upper_middle.npy', 'wb') as f:
    np.save(f, val_data_joint)

# joints in lower+middle space
train_data_joint = train_X
val_data_joint = val_X
upper_joints = [3, 4, 6, 7, 8, 10, 11, 12, 22, 23, 24, 25]
upper_joints = [k-1 for k in upper_joints]
# Zero all points in upper part, maintain lower+middle part values
for i in upper_joints:
    train_data_joint[:, :, :, i, :] = 0
    val_data_joint[:, :, :, i, :] = 0

with open('./ntu_25/xsub/train_data_joint_lower_middle.npy', 'wb') as f:
    np.save(f, train_data_joint)
with open('./ntu_25/xsub/val_data_joint_lower_middle.npy', 'wb') as f:
    np.save(f, val_data_joint)

print("")