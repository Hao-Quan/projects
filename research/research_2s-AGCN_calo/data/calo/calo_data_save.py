import numpy as np
import h5py
import pandas as pd
import pickle as pkl


f = h5py.File("data_training_test_without_shuffle.h5", 'r')
print(f.keys())

# <KeysViewHDF5 ['test_data', 'test_label', 'train_data', 'train_label']>

train_data = pd.read_hdf("data_training_test_without_shuffle.h5", "train_data").values
test_data = pd.read_hdf("data_training_test_without_shuffle.h5", "test_data").values

train_label = pd.read_hdf("data_training_test_without_shuffle.h5", "train_label").values
test_label = pd.read_hdf("data_training_test_without_shuffle.h5", "test_label").values

x = []
y = []
z = []

for i in range(len(train_data)):
    x.append([train_data[i][k] for k in range(36) if k % 2 == 0])
    y.append([train_data[i][k] for k in range(36) if k % 2 != 0])
z.append(x)
z.append(y)
z = np.array(z)
z = z.transpose(1, 0, 2)
z = np.expand_dims(z, axis=2)
z = np.expand_dims(z, axis=4)
with open('train_data_joint.npy', 'wb') as f:
    np.save(f, z)

x = []
y = []
z = []

for i in range(len(test_data)):
    x.append([test_data[i][k] for k in range(36) if k % 2 == 0])
    y.append([test_data[i][k] for k in range(36) if k % 2 != 0])
z.append(x)
z.append(y)
z = np.array(z)
z = z.transpose(1, 0, 2)
z = np.expand_dims(z, axis=2)
z = np.expand_dims(z, axis=4)
with open('test_data_joint.npy', 'wb') as f:
    np.save(f, z)

with open('train_label.pkl', 'wb') as f:
    pkl.dump(train_label, f)

with open('test_label.pkl', 'wb') as f:
    pkl.dump(test_label, f)

print("")