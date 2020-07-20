import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math

class_mapping = {"sitting_phone_talking": 10, "standing": 3, "walking_phone": 6, "walking_cart": 9,
                 "walking_fast": 2, "wandering": 7, "walking_slow": 1, "standing_phone_talking": 4,
                 "sitting": 0, "window_shopping": 5, "walking_phone_talking": 8}

# activity_class_name = "sitting"
activity_class_name = "standing"
activity_class_number = class_mapping[activity_class_name]

data = pd.read_hdf("data/calo/h5/data_training_test_without_shuffle.h5", key="train_data")
labels = pd.read_hdf("data/calo/h5/data_training_test_without_shuffle.h5", key="train_label")

labels = labels.astype('int')

idx_list_2 = np.where(labels['Y'] == activity_class_number)

#idx = 173080

for idx in range(math.floor(len(idx_list_2[0]))):
# for idx in range(2):
    print("Sample : {}".format(idx_list_2[0][idx]))
    X_train = data.iloc[idx_list_2[0][idx]]
    # X_train = data.iloc[197]
    # X_train = data.iloc[198]
    #X_train = data.iloc[230]
    axis_x = np.zeros([18])
    axis_y = np.zeros([18])

    #print("These joints are missed:")

    missed_joints = ""

    for i in range(0, 18):
        if X_train[i * 2] == -1:
            missed_joints += str(i) + "  "
            # print("{}".format(i))
        axis_x[i] = X_train[i * 2]
        axis_y[i] = X_train[i * 2 + 1]

    n = [i for i in range(18)]

    mask = axis_x != -1

    fig, ax = plt.subplots()
    ax.scatter(axis_x[mask], axis_y[mask])

    for i, txt in enumerate(n):
        ax.annotate(txt, (axis_x[i], axis_y[i]))

    # plt.plot(axis_x, axis_y, 'ro')
    plt.title("Sample: " + str(idx_list_2[0][idx]) + ", " + activity_class_name + " (" + str(activity_class_number) + "), " + "Missed joints: " + missed_joints)
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()

    plt.savefig(os.path.join("img", str(activity_class_number), str(idx_list_2[0][idx]) + ".png"))
    # plt.savefig(os.path.join("", "1_0.png"))

print("")