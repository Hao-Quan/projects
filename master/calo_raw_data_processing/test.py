import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics
import random
from random import randint
import time
import os

LABELS = [
    "JUMPING",
    "JUMPING_JACKS",
    "BOXING",
    "WAVING_2HANDS",
    "WAVING_1HAND",
    "CLAPPING_HANDS"
]

DATASET_PATH = "3_carosello/7/"

X_train_path = DATASET_PATH + "X.txt"


Y_train_path = DATASET_PATH + "Y.txt"


n_steps = 32 # 32 timesteps per series


# Load the networks inputs

def load_X(X_path):
    file = open(X_path, 'r')
    X_ = np.array(
        [elem for elem in [
            row.split(',') for row in file
        ]],
        dtype=np.float32
    )
    file.close()
    # blocks = int(len(X_) / n_steps)

    # X_ = np.array(np.split(X_, blocks))

    return X_


# Load the networks outputs

def load_y(y_path):
    file = open(y_path, 'r')
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    # for 0-based indexing
    return y_ - 1


# X_train = load_X(X_train_path)
X_train = load_X(X_train_path)
# print X_test

# y_train = load_y(y_train_path)
Y_train = load_y(Y_train_path)

axis_x = np.zeros([32, 18])
axis_y = np.zeros([32, 18])


# for j in range (0, 32):
#     for i in range(0, 18):
#         axis_x[j][i] = X_train[1][j][i * 2]
#         axis_y[j][i] = X_train[1][j][i * 2 + 1]
#     plt.plot(axis_x[j], axis_y[j], 'ro')
#     plt.show()


for j in range (0, 32):
    for i in range(0, 18):
        axis_x[j][i] = X_train[j][i * 2]
        axis_y[j][i] = X_train[j][i * 2 + 1]
    plt.plot(axis_x[j], axis_y[j], 'ro')
    plt.show()

print("C")