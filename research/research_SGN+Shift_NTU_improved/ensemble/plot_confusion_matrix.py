import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm

# plot confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

dataset = 'ntu/xsub'

def plot_confusion_matrix(target, prediction):
    x_axis_labels = [i for i in range(1, 61)]
    y_axis_labels = ["drink water 1", "eat meal/snack 2", "brushing teeth 3", " brushing hair 4", "drop 5", "pickup 6", "throw 7", "sitting down 8", "standing up (from sitting position) 9", "clapping 10",
                     "reading 11", "writing 12", "tear up paper 13", "wear jacket 14", "take off jacket 15", "wear a shoe 16", "take off a shoe 17", "wear on glasses 18", "take off glasses 19", "put on a hat/cap 20",
                     "take off a hat/cap 21", "cheer up 22", "hand waving 23", "kicking something 24", "reach into pocket 25", "hopping (one foot jumping) 26", "jump up 27", "make a phone call/answer phone 28", "playing with phone/tablet 29", "typing on a keyboard 30",
                     "pointing to something with finger 31", "taking a selfie 32", "check time (from watch) 33", "rub two hands together 34", "nod head/bow 35", "shake head 36", "wipe face 37", "salute 38", "put the palms together 39", "cross hands in front (say stop) 40",
                     "sneeze/cough 41", "staggering 42", "falling 43", "touch head (headache) 44", "touch chest (stomachache/heart pain) 45", "touch back (backache) 46", "touch neck (neckache) 47", "nausea or vomiting condition 48", "use a fan (with hand or paper)/feeling warm 49", "punching/slapping other person 50",
                     "kicking other person 51", "pushing other person 52", " pat on back of other person 53", "point finger at the other person 54", "hugging other person 55", "giving something to other person 56", "touch other person's pocket 57", "handshaking 58", "walking towards each other 59", "walking apart from each other 60"
                     ]
    # y_axis_labels = ["drink water 1", "eat meal/snack 2", "brushing teeth 3", " brushing hair 4", "drop 5", "pickup 6",
    #                  "throw 7", "sitting down 8", "standing up (from sitting position) 9", "clapping 10",
    #                  "reading 11", "writing 12", "tear up paper 13", "wear jacket 14", "take off jacket 15",
    #                  "wear a shoe 16", "take off a shoe 17", "wear on glasses 18", "take off glasses 19",
    #                  "put on a hat/cap 20",
    #                  "take off a hat/cap 21", "cheer up 22", "hand waving 23", "kicking something 24",
    #                  "reach into pocket 25", "hopping (one foot jumping) 26", "jump up 27",
    #                  "make a phone call/answer phone 28", "playing with phone/tablet 29", "typing on a keyboard 30",
    #                  "pointing to something with finger 31", "taking a selfie 32", "check time (from watch) 33",
    #                  "rub two hands together 34", "nod head/bow 35", "shake head 36", "wipe face 37", "salute 38",
    #                  "put the palms together 39", "cross hands in front (say stop) 40",
    #                  "sneeze/cough 41", "staggering 42", "falling 43", "touch head (headache) 44",
    #                  "touch chest (stomachache/heart pain) 45", "touch back (backache) 46", "touch neck (neckache) 47",
    #                  "nausea or vomiting condition 48", "use a fan (with hand or paper)/feeling warm 49",
    #                  "punching/slapping other person 50",
    #                  "kicking other person 51", "pushing other person 52", " pat on back of other person 53",
    #                  "point finger at the other person 54", "hugging other person 55",
    #                  "giving something to other person 56", "touch other person's pocket 57", "handshaking 58",
    #                  "walking towards each other 59", "walking apart from each other 60"
    #                  ]


    sns.set(font_scale=1.7)

    # Plot without normalization"

    con_mat = confusion_matrix(target, prediction, normalize='all')

    plt.figure(figsize=(90, 90))
    plotted_img = sns.heatmap(con_mat, annot=True, cmap="YlGnBu", xticklabels=x_axis_labels, yticklabels=y_axis_labels)

    for item in plotted_img.get_xticklabels():
        #item.set_rotation(45)
        item.set_size(20)

    plt.title("NTU-XSUB Confusion matrix with ALL normalization")

    plt.savefig("../images/{}/test_confusion_matrix_ROW_COLUMN_normalization.png".format(dataset))

    # Plot normalized with "target (row)"

    con_mat = confusion_matrix(target, prediction, normalize='true')

    plt.figure(figsize=(90, 90))
    plotted_img = sns.heatmap(con_mat, annot=True, cmap="YlGnBu", xticklabels=x_axis_labels, yticklabels=y_axis_labels)

    for item in plotted_img.get_xticklabels():
        #item.set_rotation(45)
        item.set_size(20)

    plt.title("NTU-XUB Confusion matrix normalized by row (target)")

    plt.savefig("../images/{}/test_confusion_matrix_target_normalizaton.png".format(dataset))

    # Plot normalized with "pred (column)"

    con_mat = confusion_matrix(target, prediction, normalize='pred')

    plt.figure(figsize=(90, 90))
    plotted_img = sns.heatmap(con_mat, annot=True, cmap="YlGnBu", xticklabels=x_axis_labels, yticklabels=y_axis_labels)

    for item in plotted_img.get_xticklabels():
        #item.set_rotation(45)
        item.set_size(20)

    plt.title("NTU-XUB Confusion matrix normalized by column (predict)")

    plt.savefig("../images/{}/test_confusion_matrix_predict_normalization.png".format(dataset))


# with open('./{}/{}_data_bone_{}.npy'.format(dataset, set, part), 'wb') as f:
#     np.save(f, store_data)

if __name__ == "__main__":
    predict_label = np.load('xsub/xsub_predict_label.npy')
    with open('../data/' + dataset + '/val_label.pkl', 'rb') as label:
        target = np.array(pickle.load(label))
    target = [int(i) for i in target[1]]
    #target = np.array(target)
    plot_confusion_matrix(target, predict_label)


