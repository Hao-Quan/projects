import numpy as np
import pickle


train_data_joint = np.load("train_data_joint.npy")
val_data_joint = np.load("val_data_joint.npy")

with open('train_label.pkl', 'rb') as f:
    label = pickle.load(f)

print("")