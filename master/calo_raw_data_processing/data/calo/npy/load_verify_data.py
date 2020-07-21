import numpy as np
import pickle as pkl

train_data = np.load('300_time_frames/train_data_joint.npy')

with open('300_time_frames/train_label.pkl', 'rb') as f:
    train_label = pkl.load(f)

print("")