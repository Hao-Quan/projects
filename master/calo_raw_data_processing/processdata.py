from os import listdir
from os.path import isfile, join

import pandas as pd
import os
from sklearn.model_selection import train_test_split
import h5py
import numpy as np

root_dir = "./"

header = []
for i in range(18):
    header.append("x_" + str(i))
    header.append("y_" + str(i))

directory_list_carosello = ["1_carosello", "3_carosello", "4_carosello", "9_carosello", "10_carosello"]
str_x = "/X.txt"
str_y = "/Y.txt"
landmarks_frame_X = pd.DataFrame(columns=header)
label_Y = pd.DataFrame(columns=["Y"])

for i in directory_list_carosello:
    path_current_carosello = root_dir + i
    p = os.walk(path_current_carosello)
    onlyfiles = [f for f in listdir(path_current_carosello) if isfile(join(path_current_carosello, f))]
    # (_, _, filenames) = os.walk(path_curent_carosello).next()
    tracese_name = []
    for (dirpath, dirnames, filenames) in os.walk(path_current_carosello):
        tracese_name.append(dirpath)
    tracese_name.sort()
    tracese_name.pop(0)

    for trace in tracese_name:
        print("Processing: " + trace + str_x)
        current_file_data_X = pd.read_csv(trace+str_x, sep=',', names=header)
        current_file_data_Y = pd.read_csv(trace + str_y, sep=',', names="Y")
        landmarks_frame_X = landmarks_frame_X.append(current_file_data_X)
        label_Y = label_Y.append(current_file_data_Y)

train_data, test_data, train_label, test_label = train_test_split(landmarks_frame_X, label_Y, test_size=0.2, shuffle=True, random_state=42)

train_data.to_hdf("data_training_test.h5", "train_data")
test_data.to_hdf("data_training_test.h5", "test_data")
train_label.to_hdf("data_training_test.h5", "train_label")
test_label.to_hdf("data_training_test.h5", "test_label")


# train_label = np.array(train_label).astype('int')
# test_label = np.array(test_label).astype('int')
# with h5py.File('data_training_test.h5', 'w') as f:
#     train_data = f.create_dataset("train_data", data=train_data)
#     train_label = f.create_dataset("train_label", data=train_label)
#     test_data = f.create_dataset("test_data", data=test_data)
#     test_label = f.create_dataset("test_label", data=test_label)


print("")