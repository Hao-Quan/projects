'''
Verify each folder
e.g. "1_carosello/0" contains one trace of video which has time relationship
     "1_carosello/1" contains another trace of video

     "1_carosello/0" and "1_carosello/1" do not have time relationship

     output image folders:
        img/1_0
        img/1_0
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math

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

class_mapping = {"sitting_phone_talking": 10, "standing": 3, "walking_phone": 6, "walking_cart": 9,
                 "walking_fast": 2, "wandering": 7, "walking_slow": 1, "standing_phone_talking": 4,
                 "sitting": 0, "window_shopping": 5, "walking_phone_talking": 8}

# activity_class_name = "sitting"
activity_class_name = "walking_slow"
activity_class_number = class_mapping[activity_class_name]

# for i in directory_list_carosello:
#     path_current_carosello = root_dir + i
#     p = os.walk(path_current_carosello)
#     onlyfiles = [f for f in listdir(path_current_carosello) if isfile(join(path_current_carosello, f))]
#     # (_, _, filenames) = os.walk(path_curent_carosello).next()
#     tracese_name = []
#     for (dirpath, dirnames, filenames) in os.walk(path_current_carosello):
#         tracese_name.append(dirpath)
#     tracese_name.sort()
#     tracese_name.pop(0)

path = ['1_carosello/1']

for trace in path:
    print("Processing: " + trace + str_x)
    current_file_data_X = pd.read_csv(trace+str_x, sep=',', names=header)
    current_file_data_Y = pd.read_csv(trace + str_y, sep=',', names="Y")
    landmarks_frame_X = landmarks_frame_X.append(current_file_data_X)
    label_Y = label_Y.append(current_file_data_Y)

    axis_x = np.zeros([18])
    axis_y = np.zeros([18])
    missed_joints = ""

    current_file_data_X = current_file_data_X.to_numpy()
    current_file_data_Y = current_file_data_Y.to_numpy()

    for t in range(0, current_file_data_X.shape[0]):
        for i in range(0, 18):
            if current_file_data_X[t][i * 2] == -1:
                missed_joints += str(i) + "  "
                # print("{}".format(i))
            axis_x[i] = current_file_data_X[t][i * 2]
            axis_y[i] = current_file_data_X[t][i * 2 + 1]

        n = [i for i in range(18)]

        mask = axis_x != -1

        fig, ax = plt.subplots()
        ax.scatter(axis_x[mask], axis_y[mask])

        for i, txt in enumerate(n):
            ax.annotate(txt, (axis_x[i], axis_y[i]))

        plt.title("Missed joints: " + missed_joints)
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()

        plt.savefig(os.path.join("img", str(activity_class_number) + "_1", str(t) + ".png"))

#train_data, test_data, train_label, test_label = train_test_split(landmarks_frame_X, label_Y, test_size=0.2, shuffle=True, random_state=42)

print("")


#
# data = pd.read_hdf("data_training_test_without_shuffle.h5", key="train_data")
# labels = pd.read_hdf("data_training_test_without_shuffle.h5", key="train_label")
#
# labels = labels.astype('int')
#
# idx_list_2 = np.where(labels['Y'] == activity_class_number)
#
# #idx = 173080
#
# for idx in range(math.floor(len(idx_list_2[0]))):
# # for idx in range(2):
#     print("Sample : {}".format(idx_list_2[0][idx]))
#     X_train = data.iloc[idx_list_2[0][idx]]
#     # X_train = data.iloc[197]
#     # X_train = data.iloc[198]
#     #X_train = data.iloc[230]
#     axis_x = np.zeros([18])
#     axis_y = np.zeros([18])
#
#     #print("These joints are missed:")
#
#     missed_joints = ""
#
#     for i in range(0, 18):
#         if X_train[i * 2] == -1_0:
#             missed_joints += str(i) + "  "
#             # print("{}".format(i))
#         axis_x[i] = X_train[i * 2]
#         axis_y[i] = X_train[i * 2 + 1_0]
#
#     n = [i for i in range(18)]
#
#     mask = axis_x != -1_0
#
#     fig, ax = plt.subplots()
#     ax.scatter(axis_x[mask], axis_y[mask])
#
#     for i, txt in enumerate(n):
#         ax.annotate(txt, (axis_x[i], axis_y[i]))
#
#     # plt.plot(axis_x, axis_y, 'ro')
#     plt.title("Sample: " + str(idx_list_2[0][idx]) + ", " + activity_class_name + " (" + str(activity_class_number) + "), " + "Missed joints: " + missed_joints)
#     plt.gca().invert_xaxis()
#     plt.gca().invert_yaxis()
#
#     plt.savefig(os.path.join("img", str(activity_class_number), str(idx_list_2[0][idx]) + ".png"))
#     # plt.savefig(os.path.join("", "1_0.png"))
#
# print("")