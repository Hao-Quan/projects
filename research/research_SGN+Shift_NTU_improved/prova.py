import numpy as np
import os

# if self.dataset == 'ntu':
#     if self.case == 0:
#         path = osp.join('data', self.dataset, 'xsub/')
#     elif self.case == 1:
#         path = osp.join('data', self.dataset, 'xview/')
#     # if self.case ==0:
#     #     self.part = 'CS'
#     # elif self.case == 1:
#     #     self.part = 'CV'
# elif self.dataset == 'calo':
#     path = osp.join('data/calo/300_time_frames/')

# Numpy for semantic (Upper + Middle partion)
#train_data = np.load("data/train_data_joint_{}_middle.npy".format(part))
#val_data = np.load("data/ntu/xsub/val_data_joint.npy")

#print(val_data.shape)
#
# path = os.readlink('data')
# os.l

# val_data = np.load("data/ntu/xsub/val_data_joint.npy")
# val_data = np.load("/home/quan/storage/data/ntu/xsub/val_data_joint.npy")
# print(val_data.shape)

# path = "/home/hao/projects/research/research_2s-AGCN_calo/data/calo/test_data_joint.npy"
# val_data = np.load(path)

for root, dirs, files in os.walk("data"):
    for filename in files:
        print(filename)

print("s")