import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm

# plot confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## !!!! REMEMBER TO CHANGE LABEL !!!! ###
    parser.add_argument('--dataset', default='ntu/xsub', choices={'calo', 'kinetics', 'ntu/xsub', 'ntu/xsub'},
                        help='the work folder for storing results')
    parser.add_argument('--alpha', default=0.5, help='weighted summation')
    parser.add_argument('--upper-dir',
                        default= './',
                        help='Directory containing "test_upper_score.pkl" for upper semantic part eval results')
    parser.add_argument('--lower-dir',
                        default='./',
                        help='Directory containing "test_lower_score.pkl" for lower semantic part eval results')

    arg = parser.parse_args()

    dataset = arg.dataset

    ## !!!! REMEMBER TO CHANGE LABEL !!!! ###

    with open('../data/' + dataset + '/val_label.pkl', 'rb') as label:
        label = np.array(pickle.load(label))

    # with open(os.path.join(arg.upper_dir, 'xsub/joint/bs_64', 'upper_test_score.pkl'), 'rb') as r1:
    #with open(os.path.join(arg.upper_dir, 'upper_test_score.pkl'), 'rb') as r1:
    with open(os.path.join(arg.upper_dir, 'xsub/joint/bs_64', 'upper_test_score.pkl'), 'rb') as r1:
        r1 = pickle.load(r1)

    with open(os.path.join(arg.lower_dir, 'xsub/joint/bs_64', 'lower_test_score.pkl'), 'rb') as r2:
    #with open(os.path.join(arg.lower_dir, 'lower_test_score.pkl'), 'rb') as r2:
        r2 = pickle.load(r2)

    # 2 streams

    with open(os.path.join(arg.upper_dir, 'xsub/bone', 'upper_test_score.pkl'), 'rb') as r3:
    #with open(os.path.join(arg.upper_dir, 'upper_test_score.pkl'), 'rb') as r1:
        r3 = pickle.load(r3)

    with open(os.path.join(arg.lower_dir, 'xsub/bone', 'lower_test_score.pkl'), 'rb') as r4:
    #with open(os.path.join(arg.lower_dir, 'lower_test_score.pkl'), 'rb') as r2:
        r4 = pickle.load(r4)

    # 4 streams
    #joint motion
    with open(os.path.join(arg.upper_dir, 'xsub/joint_motion', 'upper_test_score.pkl'), 'rb') as r5:
    #with open(os.path.join(arg.upper_dir, 'upper_test_score.pkl'), 'rb') as r1:
        r5 = pickle.load(r5)

    with open(os.path.join(arg.lower_dir, 'xsub/joint_motion', 'lower_test_score.pkl'), 'rb') as r6:
    #with open(os.path.join(arg.lower_dir, 'lower_test_score.pkl'), 'rb') as r2:
        r6 = pickle.load(r6)

    #bone motion
    with open(os.path.join(arg.upper_dir, 'xsub/bone_motion', 'upper_test_score.pkl'), 'rb') as r7:
    #with open(os.path.join(arg.upper_dir, 'upper_test_score.pkl'), 'rb') as r1:
        r7 = pickle.load(r7)

    with open(os.path.join(arg.lower_dir, 'xsub/joint_motion', 'lower_test_score.pkl'), 'rb') as r8:
    #with open(os.path.join(arg.lower_dir, 'lower_test_score.pkl'), 'rb') as r2:
        r8 = pickle.load(r8)

    right_num = total_num = right_num_5 = 0

    # for Confusion Matrix
    list_pred = []

    for i in tqdm(range(len(label[0]))):
        _, l = label[:, i]

        #joint
        _, r11 = list(r1.items())[i]
        _, r22 = list(r2.items())[i]
        r_joint = r11 + r22 * arg.alpha
        #bone
        _, r33 = list(r3.items())[i]
        _, r44 = list(r4.items())[i]
        r_bone = r33 + r44 * arg.alpha
        #joint_motion
        _, r55 = list(r5.items())[i]
        _, r66 = list(r6.items())[i]
        r_joint_motion = r55 + r66 * arg.alpha
        # bone_motion
        _, r77 = list(r7.items())[i]
        _, r88 = list(r8.items())[i]
        r_bone_motion = r77 + r88 * arg.alpha

        r = r_joint + r_bone + r_joint_motion + r_bone_motion

        #r = r11
        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1

        # append predict label in list for Confusion matrix
        list_pred.append(r)

    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))

    with open('./xsub/xsub_predict_label.npy', 'wb') as f:
        print("save ./xsub/xsub_predict_label.npy")
        np.save(f, list_pred)

    #train_X = np.load('ntu_light/xsub/train_data_joint_motion.npy', mmap_mode='r')


def plot_confusion_matrix(target, prediction):
    x_axis_labels = ["sitting", "walking_slow", "walking_fast", "standing", "standing_phone_talking", "window_shopping", "walking_phone", "wandering", "walking_phone_talking", "walking_cart", "sitting_phone_talking"]

    y_axis_labels = ["sitting", "walking_slow", "walking_fast", "standing", "standing_phone_talking", "window_shopping", "walking_phone", "wandering", "walking_phone_talking", "walking_cart", "sitting_phone_talking"]

    sns.set(font_scale=1.7)

    # Plot without normalization"

    con_mat = confusion_matrix(target, prediction)

    plt.figure(figsize=(30, 25))
    plotted_img = sns.heatmap(con_mat, annot=True, cmap="YlGnBu", xticklabels=x_axis_labels, yticklabels=y_axis_labels)

    for item in plotted_img.get_xticklabels():
        item.set_rotation(45)
        item.set_size(20)

    plt.title("Confusion matrix without normalization")

    plt.savefig("images/test_confusion_matrix_without_normalization.png")

    # Plot normalized with "target (row)"

    con_mat = confusion_matrix(target, prediction, normalize='true')

    plt.figure(figsize=(30, 25))
    plotted_img = sns.heatmap(con_mat, annot=True, cmap="YlGnBu", xticklabels=x_axis_labels, yticklabels=y_axis_labels)

    for item in plotted_img.get_xticklabels():
        item.set_rotation(45)
        item.set_size(20)

    plt.title("Confusion matrix normalized by row (target)")

    plt.savefig("images/test_confusion_matrix_target.png")

    # Plot normalized with "pred (column)"

    con_mat = confusion_matrix(target, prediction, normalize='pred')

    plt.figure(figsize=(30, 25))
    plotted_img = sns.heatmap(con_mat, annot=True, cmap="YlGnBu", xticklabels=x_axis_labels, yticklabels=y_axis_labels)

    for item in plotted_img.get_xticklabels():
        item.set_rotation(45)
        item.set_size(20)

    plt.title("Confusion matrix normalized by column (predict)")

    plt.savefig("images/test_confusion_matrix_pred.png")


# with open('./{}/{}_data_bone_{}.npy'.format(dataset, set, part), 'wb') as f:
#     np.save(f, store_data)