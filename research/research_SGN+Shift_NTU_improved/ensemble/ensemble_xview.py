import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## !!!! REMEMBER TO CHANGE LABEL !!!! ###
    parser.add_argument('--dataset', default='ntu/xview', choices={'calo', 'kinetics', 'ntu/xview', 'ntu/xview'},
                        help='the work folder for storing results')
    parser.add_argument('--alpha', default=0.5, help='weighted summation')
    parser.add_argument('--upper-dir',
                        default= '.',
                        help='Directory containing "test_upper_score.pkl" for upper semantic part eval results')
    parser.add_argument('--lower-dir',
                        default='.',
                        help='Directory containing "test_lower_score.pkl" for lower semantic part eval results')

    arg = parser.parse_args()

    dataset = arg.dataset

    ## !!!! REMEMBER TO CHANGE LABEL !!!! ###

    with open('../data/' + dataset + '/val_label.pkl', 'rb') as label:
        label = np.array(pickle.load(label))

    # with open(os.path.join(arg.upper_dir, 'xview/bone', 'upper_test_score.pkl'), 'rb') as r1:
    #with open(os.path.join(arg.upper_dir, 'upper_test_score.pkl'), 'rb') as r1:
    with open(os.path.join(arg.upper_dir, 'xview/joint_motion', 'upper_test_score.pkl'), 'rb') as r1:
        r1 = pickle.load(r1)

    # with open(os.path.join(arg.lower_dir, 'xview/bone', 'lower_test_score.pkl'), 'rb') as r2:
    with open(os.path.join(arg.lower_dir, 'xview/joint_motion', 'lower_test_score.pkl'), 'rb') as r2:
    #with open(os.path.join(arg.lower_dir, 'lower_test_score.pkl'), 'rb') as r2:
        r2 = pickle.load(r2)

    right_num = total_num = right_num_5 = 0
    for i in tqdm(range(len(label[0]))):
        _, l = label[:, i]
        _, r11 = list(r1.items())[i]
        _, r22 = list(r2.items())[i]
        r = r11 + r22 * arg.alpha
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
