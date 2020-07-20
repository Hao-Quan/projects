import numpy as np
import pickle as pkl

train_data = np.load('calo_300_without_padding.npy')

with open('train_label.pkl', 'rb') as f:
    train_label = pkl.load(f)

print("")