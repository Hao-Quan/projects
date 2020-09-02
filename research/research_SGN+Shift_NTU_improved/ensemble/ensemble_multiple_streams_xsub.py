import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm

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
    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))

    # right_num = total_num = right_num_5 = 0
    # for i in tqdm(range(len(label))):
    #     l = label[i]
    #     r11 = r1[i]
    #     r22 = r2[i]
    #     r = r11 + r22 * arg.alpha
    #     rank_5 = r.argsort()[-5:]
    #     right_num_5 += int(int(l) in rank_5)
    #     r = np.argmax(r)
    #     right_num += int(r == int(l))
    #     total_num += 1
    # acc = right_num / total_num
    # acc5 = right_num_5 / total_num
    #
    # print('Top1 Acc: {:.4f}%'.format(acc * 100))
    # print('Top5 Acc: {:.4f}%'.format(acc5 * 100))

# import argparse
# import pickle

# import numpy as np
# from tqdm import tqdm
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--datasets', default='calo', choices={'calo', 'kinetics', 'ntu/xsub', 'ntu/xsub'},
#                     help='the work folder for storing results')
# parser.add_argument('--alpha', default=1, help='weighted summation')
# arg = parser.parse_args()
#
# dataset = arg.datasets
# label = open('./data/' + dataset + '/val_label.pkl', 'rb')
# label = np.array(pickle.load(label))
# r1 = open('./work_dir/' + dataset + '/agcn_test_joint/epoch1_test_score.pkl', 'rb')
# r1 = list(pickle.load(r1).items())
# r2 = open('./work_dir/' + dataset + '/agcn_test_bone/epoch1_test_score.pkl', 'rb')
# r2 = list(pickle.load(r2).items())
# right_num = total_num = right_num_5 = 0
# for i in tqdm(range(len(label[0]))):
#     _, l = label[:, i]
#     _, r11 = r1[i]
#     _, r22 = r2[i]
#     r = r11 + r22 * arg.alpha
#     rank_5 = r.argsort()[-5:]
#     right_num_5 += int(int(l) in rank_5)
#     r = np.argmax(r)
#     right_num += int(r == int(l))
#     total_num += 1
# acc = right_num / total_num
# acc5 = right_num_5 / total_num
# print(acc, acc5)
