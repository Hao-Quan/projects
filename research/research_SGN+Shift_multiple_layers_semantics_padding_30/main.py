# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
import time
import shutil
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import os.path as osp
import csv
import numpy as np

np.random.seed(1337)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from model import SGN
from data import CaloDataLoaders, AverageMeter
import fit
from util import make_dir, get_num_classes

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl

import yaml

# parser = argparse.ArgumentParser(description='Skeleton-Based Action Recgnition')
# fit.add_fit_args(parser)
# parser.set_defaults(
#     network='SGN',
#     dataset='calo',
#     metric='upper',
#     case=0,
#     batch_size=32,
#     max_epochs=100,
#     monitor='val_acc',
#     lr=0.001,
#     weight_decay=0.0001,
#     lr_factor=0.1,
#     workers=16,
#     print_freq = 20,
#     train = 1,
#     seg = 20,
#     graph = 'graph.calo.Graph'
#     )
# args = parser.parse_args()

def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Sematic ShiftGraph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/calo/test_lower_middle.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=True,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=2,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=0,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--seg',
        type=int,
        default=1,
        help='segment')
    parser.add_argument(
        '--metric',
        type=str,
        default='upper',
        help='upper or lower semantic space')
    parser.add_argument(
        '--graph',
        type=str,
        default='calo',
        help='adjacency matrix graph for specific dataset'
    )
    parser.add_argument(
        '--monitor',
        type=str,
        default='acc_val',
        help='monitor validation accuracy'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='calo',
        help='selection dataset'
    )
    parser.add_argument(
        '--case',
        type=int,
        default=0,
        help='cross subject / cross view'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=0,
        help='workers number for dataloader'
    )
    parser.add_argument(
        '--train',
        type=int,
        default=1,
        help='train or test phase'
    )
    parser.add_argument(
        '--save_score',
        type=bool,
        default=True,
        help='save test score'
    )

    parser.add_argument('--only_train_part', default=False)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)
    return parser

class Processor():
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()


    def start(self):
        model = SGN(args.model_args['num_class'], args.model_args['seg'], args, graph=args.graph)
        # args.num_classes = get_num_classes(args.dataset)
        # model = SGN(args.num_classes, args.dataset, args.seg, args, graph=args.graph)

        total = self.get_n_params(model)
        print(model)
        print('The number of parameters: ', total)
        #print('The modes is:', args.network)

        if torch.cuda.is_available():
            print('It is using GPU!')
            model = model.cuda()

        criterion = LabelSmoothingLoss(args.model_args['num_class'], smoothing=0.1).cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)

        if args.monitor == 'val_acc':
            mode = 'max'
            monitor_op = np.greater
            best = -np.Inf
            str_op = 'improve'
        elif args.monitor == 'val_loss':
            mode = 'min'
            monitor_op = np.less
            best = np.Inf
            str_op = 'reduce'

        scheduler = MultiStepLR(optimizer, milestones=[60, 90, 110], gamma=0.1)
        # Data loading
        calo_loaders = CaloDataLoaders(args.dataset, args.metric, args.case, seg=args.seg)
        train_loader = calo_loaders.get_train_loader(args.batch_size, args.workers)
        #val_loader = ntu_loaders.get_val_loader(args.batch_size, args.workers)
        train_size = calo_loaders.get_train_size()
        val_size = calo_loaders.get_val_size()
        #val_size = ntu_loaders.get_val_size()

        val_loader = calo_loaders.get_val_loader(32, args.workers)

        # print('Train on %d samples, validate on %d samples' % (train_size, val_size))
        print('Train on %d samples, test on %d X samples' % (train_size, val_size))

        best_epoch = 0
        output_dir = make_dir(args.dataset)

        save_path = os.path.join(output_dir)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        checkpoint = osp.join(save_path, '%s_best.pth' % args.metric)
        earlystop_cnt = 0
        csv_file = osp.join(save_path, '%s_log.csv' % args.metric)
        log_res = list()

        lable_path = osp.join(save_path, '%s_lable.txt'% args.metric)
        pred_path = osp.join(save_path, '%s_pred.txt' % args.metric)

        # Training
        if args.phase == 'train':
            for epoch in range(args.start_epoch, args.num_epoch):

                print('Epoch: ', epoch, optimizer.param_groups[0]['lr'])

                t_start = time.time()
                train_loss, train_acc = self.train(train_loader, model, criterion, optimizer, epoch)

                test_loss, val_acc = self.validate(val_loader, model, criterion)
                log_res += [[train_loss, train_acc.cpu().numpy(), \
                             test_loss, val_acc.cpu().numpy()]]

                print('Epoch-{:<3d} {:.1f}s\t'
                      'Train: loss {:.4f}\taccu {:.4f}\tValid: loss {:.4f}\taccu {:.4f}'
                      .format(epoch + 1, time.time() - t_start, train_loss, train_acc, test_loss, val_acc))

                current = test_loss if mode == 'min' else val_acc

                # # Original version with VALIDATION
                # val_loss, val_acc = validate(val_loader, model, criterion)
                # log_res += [[train_loss, train_acc.cpu().numpy(),\
                #              val_loss, val_acc.cpu().numpy()]]
                #
                # print('Epoch-{:<3d} {:.1f}s\t'
                #       'Train: loss {:.4f}\taccu {:.4f}\tValid: loss {:.4f}\taccu {:.4f}'
                #       .format(epoch + 1, time.time() - t_start, train_loss, train_acc, val_loss, val_acc))
                #
                # current = val_loss if mode == 'min' else val_acc

                ####### store tensor in cpu
                current = current.cpu()

                if monitor_op(current, best):
                    print('Epoch %d: %s %sd from %.4f to %.4f, '
                          'saving model to %s'
                          % (epoch + 1, args.monitor, str_op, best, current, checkpoint))
                    best = current
                    best_epoch = epoch + 1
                    self.save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best': best,
                        'monitor': args.monitor,
                        'optimizer': optimizer.state_dict(),
                    }, checkpoint)
                    earlystop_cnt = 0
                else:
                    print('Epoch %d: %s did not %s' % (epoch + 1, args.monitor, str_op))
                    earlystop_cnt += 1

                scheduler.step()

            print('Best %s: %.4f from epoch-%d' % (args.monitor, best, best_epoch))
            with open(csv_file, 'w') as fw:
                cw = csv.writer(fw)
                cw.writerow(['loss', 'acc', 'val_loss', 'val_acc'])
                cw.writerows(log_res)
            print('Save train and validation log into into %s' % csv_file)

            # After Training phase, now run Test phase
            args.train = 0
            model = SGN(args.model_args['num_class'], args.model_args['seg'], args, graph=args.graph)
            model = model.cuda()
            self.test(val_loader, model, checkpoint, lable_path, pred_path)

        # Only run Test
        else:
            weights = torch.load(self.arg.weights)
            model.load_state_dict(weights['state_dict'])
            model = model.cuda()
            self.test(val_loader, model, checkpoint, lable_path, pred_path)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)


    def train(self, train_loader, model, criterion, optimizer, epoch):
        losses = AverageMeter()
        acces = AverageMeter()
        model.train()

        for i, (inputs, target) in enumerate(train_loader):     #train_loader.dataset[x]: (35673 x 300 x 150); train_loader.dataset[y]: (35673)
            inputs = inputs.float()
            output = model(inputs.cuda())   # inputs: torch.Size([64, 20, 75])  -- [batch_size X #segments? X (75=25x3)]; outputs: [batch_size X #classes(60)]
            target = target.cuda()          # target: [batch_size]
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc = self.accuracy(output.data, target)
            losses.update(loss.item(), inputs.size(0))
            acces.update(acc[0], inputs.size(0))

            # backward
            optimizer.zero_grad()  # clear gradients out before each mini-batch
            loss.backward()
            optimizer.step()

            if (i + 1) % 20 == 0:
                print('Epoch-{:<3d} {:3d} batches\t'
                      'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'accu {acc.val:.3f} ({acc.avg:.3f})'.format(
                       epoch + 1, i + 1, loss=losses, acc=acces))

        return losses.avg, acces.avg


    def validate(self, val_loader, model, criterion):
        losses = AverageMeter()
        acces = AverageMeter()
        model.eval()

        for i, (inputs, target) in enumerate(val_loader):
            inputs = inputs.float()
            with torch.no_grad():
                output = model(inputs.cuda())
            target = target.cuda()
            with torch.no_grad():
                loss = criterion(output, target)

            # measure accuracy and record loss
            acc = self.accuracy(output.data, target)
            losses.update(loss.item(), inputs.size(0))
            acces.update(acc[0], inputs.size(0))

        return losses.avg, acces.avg


    def test(self, test_loader, model, checkpoint, lable_path, pred_path):
        acces = AverageMeter()
        # load learnt model that obtained best performance on validation set
        model.load_state_dict(torch.load(checkpoint)['state_dict'])
        model.eval()

        label_output = list()
        pred_output = list()
        target_final_result = list()
        pred_final_result = list()
        score_frag = []

        t_start = time.time()
        for i, (inputs, target) in enumerate(test_loader):
            inputs = inputs.float()
            with torch.no_grad():
                output = model(inputs.cuda())
                score_frag.append(output.data.cpu().numpy())
                output = output.view((-1, inputs.size(0)//target.size(0), output.size(1)))
                output = output.mean(1)

            label_output.append(target.cpu().numpy())
            pred_output.append(output.cpu().numpy())

            acc = self.accuracy_withlist(output.data, target.cuda(), pred_final_result, target_final_result)
            #acc = accuracy(output.data, target.cuda())
            #acces.update(acc[0], inputs.size(0))
        score = np.concatenate(score_frag)
        score_dict = score  # score.shape = (43829, 11)
        # print_log('\tMean {} loss of {} batches: {}.'.format(
        #     ln, len(self.data_loader[ln]), np.mean(loss_value)))
        # for k in self.arg.show_topk:
        #     self.print_log('\tTop{}: {:.2f}%'.format(
        #         k, 100 * self.data_loader[ln].dataset.top_k(score, k)))
        # save_score = True
        if self.arg.save_score:
            with open('{}/test_{}_score.pkl'.format(
                    self.arg.work_dir, self.arg.metric), 'wb') as f:
                pkl.dump(score_dict, f)
            print("Save 'test_{}_score.pkl' into {} ".format(
                self.arg.metric, self.arg.work_dir))
        # prev = pred_final_result[0]
        # for i in range(1, len(pred_final_result)):
        #     prev = torch.cat((prev, pred_final_result[i]), axis=0)

        prev = pred_final_result[0].cpu().detach().numpy()
        for i in range(1, len(pred_final_result)):
            prev = np.concatenate((prev, pred_final_result[i].cpu().detach().numpy()), axis=1)
        prev = np.squeeze(prev)

        targ = target_final_result[0].cpu().detach().numpy()
        for i in range(1, len(target_final_result)):
            targ = np.concatenate((targ, target_final_result[i].cpu().detach().numpy()), axis=0)
        targ = np.squeeze(targ).astype('int')

        list_index_correct = np.where(prev == targ)
        test_accuracy = len(list_index_correct[0]) / len(targ) * 100


        label_output = np.concatenate(label_output, axis=0)
        np.savetxt(lable_path, label_output, fmt='%d')
        pred_output = np.concatenate(pred_output, axis=0)
        np.savetxt(pred_path, targ, fmt='%d')

        # print('Test: accuracy {:.3f}, time: {:.2f}s'
        #       .format(acces.avg, time.time() - t_start))

        print('My test: accuracy {:.3f}'
              .format(test_accuracy))


        #self.plot_confusion_matrix(label_output, prev)

    def plot_confusion_matrix(self, target, prediction):
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

    def accuracy(self, output, target):
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct = correct.view(-1).float().sum(0, keepdim=True)

        return correct.mul_(100.0 / batch_size)

    def accuracy_withlist(self, output, target, pred_final_list, target_final_list):
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct = correct.view(-1).float().sum(0, keepdim=True)

        target_final_list.append(target)
        pred_final_list.append(pred)

        return correct.mul_(100.0 / batch_size)

    def save_checkpoint(self, state, filename='checkpoint.pth.tar', is_best=False):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')

    def get_n_params(self, model):
        pp=0
        for p in list(model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open(os.path.join(self.arg.work_dir, self.arg.metric + '_config.yaml'), 'w') as f:
            yaml.dump(arg_dict, f)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    args = parser.parse_args()
    processor = Processor(args)
    processor.start()
    
