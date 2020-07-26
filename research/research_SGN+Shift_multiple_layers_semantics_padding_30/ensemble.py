import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='calo', choices={'calo', 'kinetics', 'ntu/xsub', 'ntu/xview'},
                        help='the work folder for storing results')
    parser.add_argument('--alpha', default=1, help='weighted summation')
    parser.add_argument('--upper-dir',
                        default= './eval/calo/upper_middle',
                        help='Directory containing "test_upper_score.pkl" for upper semantic part eval results')
    parser.add_argument('--lower-dir',
                        default='./eval/calo/lower_middle',
                        help='Directory containing "test_lower_score.pkl" for lower semantic part eval results')

    arg = parser.parse_args()

    dataset = arg.dataset

    with open('./data/' + dataset + '/val_label.pkl', 'rb') as label:
        label = np.array(pickle.load(label))

    with open(os.path.join(arg.upper_dir, 'test_upper_score.pkl'), 'rb') as r1:
    #with open('./eval/' + dataset + '/test_upper_score.pkl', 'rb') as r1:
        r1 = pickle.load(r1)

    with open(os.path.join(arg.lower_dir, 'test_lower_score.pkl'), 'rb') as r2:
    #with open('./eval/' + dataset + '/test_lower_score.pkl', 'rb') as r2:
        r2 = pickle.load(r2)

    right_num = total_num = right_num_5 = 0
    for i in tqdm(range(len(label))):
        l = label[i]
        r11 = r1[i]
        r22 = r2[i]
        r = r11 + r22 * arg.alpha
        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
    acc = right_num / total_num
    acc5 = right_num_5 / total_num

    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))

# import argparse
# import pickle

# import numpy as np
# from tqdm import tqdm
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--datasets', default='calo', choices={'calo', 'kinetics', 'ntu/xsub', 'ntu/xview'},
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
