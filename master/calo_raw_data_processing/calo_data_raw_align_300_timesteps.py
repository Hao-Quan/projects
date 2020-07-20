from os import listdir
from os.path import isfile, join

import pandas as pd
import os
from sklearn.model_selection import train_test_split
import h5py
import numpy as np
import pickle
from data_gen.preprocess import pre_normalization
root_dir = "./"

load_data_test = True
process_train_data = False
# Total folders number 1384
# "1_carosello": 178
# "3_carosello": 402
# "4_carosello": 469
# "9_carosello": 226
# "10_carosello": 109

# We select {1, 3, 4, 10}_carosello as TRAINING data: 1158 = 178+402+469+109
#           9_carosello as VAL data: 226

if load_data_test == False:
    max_frame = 300
    num_joint = 18
    max_body_true = 1

    header = []
    for i in range(18):
        header.append("x_" + str(i))
        header.append("y_" + str(i))

    if process_train_data == True:
        directory_list_carosello = ["1_carosello", "3_carosello", "4_carosello", "10_carosello"]
        path_data = 'data/calo/npy/train_data_joint.npy'
        path_label = 'data/calo/npy/train_label.pkl'
        dimen_array = 1158
    else:
        directory_list_carosello = ["9_carosello"]
        path_data = 'data/calo/npy/val_data_joint.npy'
        path_label = 'data/calo/npy/val_label.pkl'
        dimen_array = 226
    str_x = "/X.txt"
    str_y = "/Y.txt"
    landmarks_frame_X = pd.DataFrame(columns=header)
    label_Y = pd.DataFrame(columns=["Y"])

    axis_x = np.zeros([18])
    axis_y = np.zeros([18])

    count = 0
    sample_name = []
    sample_label = []

    fp = np.zeros((dimen_array, 2, max_frame, num_joint, max_body_true), dtype=np.float32)

    for i in directory_list_carosello:
        path_current_carosello = root_dir + i
        p = os.walk(path_current_carosello)
        onlyfiles = [f for f in listdir(path_current_carosello) if isfile(join(path_current_carosello, f))]
        tracese_name = []
        for (dirpath, dirnames, filenames) in os.walk(path_current_carosello):
            tracese_name.append(dirpath)
        tracese_name.sort()
        tracese_name.pop(0)

        for trace in tracese_name:
            print("Processing: " + trace + str_x)
            #

            current_file_data_X = pd.read_csv(trace+str_x, sep=',', names=header)
            current_file_data_Y = pd.read_csv(trace + str_y, sep=',', names="Y")

            current_file_data_X = current_file_data_X.to_numpy()
            current_file_data_Y = current_file_data_Y.to_numpy()
            current_label = current_file_data_Y[0].item()

            list_total_frame = []

            for t in range(0, current_file_data_X.shape[0]):
                for i in range(0, 18):
                    axis_x[i] = current_file_data_X[t][i * 2]
                    axis_y[i] = current_file_data_X[t][i * 2 + 1]
                one_frame = np.stack((axis_x, axis_y), axis=0)
                list_total_frame.append(one_frame)
            data = np.stack(list_total_frame)
            data = data.transpose(1, 0, 2)
            data = np.expand_dims(data, 3)

            if data.shape[1] <= max_frame:
                fp[count, :, 0:data.shape[1], :, :] = data
                sample_name.append(trace[2:])
                sample_label.append(current_label)
            else:
                fp[count, :, 0:max_frame, :, :] = data[:, :max_frame, :]
                count += 1
                sample_name.append(trace[2:])
                sample_label.append(current_label)
                times = int(np.floor(data.shape[1] // max_frame))
                rest = data.shape[1] % max_frame
                for m in range(1, times):
                    fp[count, :, 0:max_frame * m, :, :] = data[:, max_frame * m : max_frame*(m+1), :]
                    count += 1
                    sample_name.append(trace[2:])
                    sample_label.append(current_label)
                if rest != 0:
                    fp[count, :, :rest, :, :] = data[:, max_frame * times: , :]
                    sample_name.append(trace[2:])
                    sample_label.append(current_label)

            count += 1
        print("end one cycle carosello")
    fp = pre_normalization(fp)

    np.save(path_data, fp)

    with open(path_label, 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

else:
    fp = np.load('data/calo/npy/train_data_joint.npy')
    with open('data/calo/npy/train_label.pkl', 'rb') as f:
        train_label = pickle.load(f)

    val_data = np.load('data/calo/npy/val_data_joint.npy')
    with open('data/calo/npy/val_label.pkl', 'rb') as f:
        val_label = pickle.load(f)

    #fp = pre_normalization(fp)
    #train_data, test_data, train_label, test_label = train_test_split(fp, train_label, test_size=0.2, shuffle=False, random_state=42)
    #np.save('data/calo/npy/train_data_joint_without_shuffle.npy', train_data)
print("")