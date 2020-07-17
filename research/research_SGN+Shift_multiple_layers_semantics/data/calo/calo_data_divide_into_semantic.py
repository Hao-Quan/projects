import numpy as np
import pickle as pkl

train_data = np.load("train_data_joint.npy")
test_data = np.load("test_data_joint.npy")

with open('train_label.pkl', 'rb') as f:
    train_label = pkl.load(f)

with open('test_label.pkl', 'rb') as f:
    test_label = pkl.load(f)


print("")