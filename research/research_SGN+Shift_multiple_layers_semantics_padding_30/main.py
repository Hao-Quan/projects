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


parser = argparse.ArgumentParser(description='Skeleton-Based Action Recgnition')
fit.add_fit_args(parser)
parser.set_defaults(
    network='SGN',
    #dataset = 'NTU',
    dataset = 'calo',
    case = 0,
    batch_size=32,
    max_epochs=100,
    monitor='val_acc',
    lr=0.001,
    weight_decay=0.0001,
    lr_factor=0.1,
    workers=16,
    print_freq = 20,
    train = 1,
    seg = 20,
    graph = 'graph.calo.Graph'
    )
args = parser.parse_args()

def main():

    args.num_classes = get_num_classes(args.dataset)
    model = SGN(args.num_classes, args.dataset, args.seg, args, graph=args.graph)

    total = get_n_params(model)
    print(model)
    print('The number of parameters: ', total)
    print('The modes is:', args.network)

    if torch.cuda.is_available():
        print('It is using GPU!')
        model = model.cuda()

    criterion = LabelSmoothingLoss(args.num_classes, smoothing=0.1).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
    calo_loaders = CaloDataLoaders(args.dataset, args.case, seg=args.seg)
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

    save_path = os.path.join(output_dir, args.network)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    checkpoint = osp.join(save_path, '%s_best.pth' % args.case)
    earlystop_cnt = 0
    csv_file = osp.join(save_path, '%s_log.csv' % args.case)
    log_res = list()

    lable_path = osp.join(save_path, '%s_lable.txt'% args.case)
    pred_path = osp.join(save_path, '%s_pred.txt' % args.case)

    # Training
    if args.train ==1:
        for epoch in range(args.start_epoch, args.max_epochs):

            print(epoch, optimizer.param_groups[0]['lr'])

            t_start = time.time()
            train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)

            test_loss, val_acc = validate(val_loader, model, criterion)
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
                save_checkpoint({
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

    ### Test
    args.train = 0
    model = SGN(args.num_classes, args.dataset, args.seg, args)
    model = model.cuda()
    test(val_loader, model, checkpoint, lable_path, pred_path)



def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    acces = AverageMeter()
    model.train()

    for i, (inputs, target) in enumerate(train_loader):     #train_loader.dataset[x]: (35673 x 300 x 150); train_loader.dataset[y]: (35673)
        inputs = inputs.float()
        output = model(inputs.cuda())   # inputs: torch.Size([64, 20, 75])  -- [batch_size X #segments? X (75=25x3)]; outputs: [batch_size X #classes(60)]
        target = target.cuda()          # target: [batch_size]
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc = accuracy(output.data, target)
        losses.update(loss.item(), inputs.size(0))
        acces.update(acc[0], inputs.size(0))

        # backward
        optimizer.zero_grad()  # clear gradients out before each mini-batch
        loss.backward()
        optimizer.step()

        if (i + 1) % args.print_freq == 0:
            print('Epoch-{:<3d} {:3d} batches\t'
                  'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'accu {acc.val:.3f} ({acc.avg:.3f})'.format(
                   epoch + 1, i + 1, loss=losses, acc=acces))

    return losses.avg, acces.avg


def validate(val_loader, model, criterion):
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
        acc = accuracy(output.data, target)
        losses.update(loss.item(), inputs.size(0))
        acces.update(acc[0], inputs.size(0))

    return losses.avg, acces.avg


def test(test_loader, model, checkpoint, lable_path, pred_path):
    acces = AverageMeter()
    # load learnt model that obtained best performance on validation set
    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    model.eval()

    label_output = list()
    pred_output = list()
    target_final_result = list()
    pred_final_result = list()


    t_start = time.time()
    for i, (inputs, target) in enumerate(test_loader):
        inputs = inputs.float()
        with torch.no_grad():
            output = model(inputs.cuda())
            output = output.view((-1, inputs.size(0)//target.size(0), output.size(1)))
            output = output.mean(1)

        label_output.append(target.cpu().numpy())
        pred_output.append(output.cpu().numpy())

        acc = accuracy_withlist(output.data, target.cuda(), pred_final_result, target_final_result)
        #acc = accuracy(output.data, target.cuda())
        #acces.update(acc[0], inputs.size(0))

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

    print('Test: accuracy {:.3f}, time: {:.2f}s'
          .format(acces.avg, time.time() - t_start))

    print('My test: accuracy {:.3f}'
          .format(test_accuracy))

    plot_confusion_matrix(label_output, prev)

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

def accuracy(output, target):
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = correct.view(-1).float().sum(0, keepdim=True)

    return correct.mul_(100.0 / batch_size)

def accuracy_withlist(output, target, pred_final_list, target_final_list):
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = correct.view(-1).float().sum(0, keepdim=True)

    target_final_list.append(target)
    pred_final_list.append(pred)

    return correct.mul_(100.0 / batch_size)

def save_checkpoint(state, filename='checkpoint.pth.tar', is_best=False):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

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


if __name__ == '__main__':
    main()
    
