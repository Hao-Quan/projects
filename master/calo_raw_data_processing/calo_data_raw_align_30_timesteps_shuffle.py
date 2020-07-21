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

load_data_test = False
# Total folders number 30 time frames:
# 7872 = 924+2384+2699+1263+602
# "1_carosello": 924
# "3_carosello": 2384
# "4_carosello": 2699
# "9_carosello": 1263
# "10_carosello": 602

dimen_array = 7872

if load_data_test == False:
    max_frame = 30
    num_joint = 18
    max_body_true = 1

    header = []
    for i in range(18):
        header.append("x_" + str(i))
        header.append("y_" + str(i))


    directory_list_carosello = ["1_carosello", "3_carosello", "4_carosello", "9_carosello", "10_carosello"]
    #directory_list_carosello = ["9_carosello"]
    path_data = 'data/calo/npy/30_time_frames/train_data_joint.npy'
    path_label = 'data/calo/npy/30_time_frames/train_label.pkl'

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
                count += 1
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
                    count += 1
                    sample_name.append(trace[2:])
                    sample_label.append(current_label)
        print("end one cycle carosello")
    fp = pre_normalization(fp)

    train_data, val_data, train_label, val_label = train_test_split(fp, sample_label,
                                                                                    test_size=0.2,
                                                                                    shuffle=True,
                                                                                    random_state=42)
    # indices = range(dimen_array)
    # train_data, val_data, train_label, val_label, indices_train, indices_val = train_test_split(fp, sample_label, indices, test_size=0.2, shuffle=True, random_state=42)

    print("")
    # np.save(path_data, fp)
    #
    # with open(path_label, 'wb') as f:
    #     pickle.dump((sample_name, list(sample_label)), f)

    np.save(path_data, train_data)
    with open(path_label, 'wb') as f:
        pickle.dump(list(train_label), f)

    np.save('data/calo/npy/30_time_frames/val_data_joint', val_data)
    with open('data/calo/npy/30_time_frames/val_label.pkl', 'wb') as f:
        pickle.dump(list(val_label), f)

else:
    fp = np.load('data/calo/npy/30_time_frames/train_data_joint.npy')
    with open('data/calo/npy/30_time_frames/train_label.pkl', 'rb') as f:
        train_label = pickle.load(f)

    val_data = np.load('data/calo/npy/30_time_frames/val_data_joint.npy')
    with open('data/calo/npy/30_time_frames/val_label.pkl', 'rb') as f:
        val_label = pickle.load(f)

    #fp = pre_normalization(fp)
    #train_data, test_data, train_label, test_label = train_test_split(fp, train_label, test_size=0.2, shuffle=False, random_state=42)
    #np.save('data/calo/npy/train_data_joint_without_shuffle.npy', train_data)
print("")