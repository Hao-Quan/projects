# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
import time
import shutil
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import os.path as osp
import csv
import numpy as np
import inspect
from collections import OrderedDict

np.random.seed(1337)
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
#from model.semantic_shift import SGN
from data import CaloDataLoaders, AverageMeter
from util import make_dir
from utils import count_params, import_class
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl
import pprint
from torch.autograd import Variable

import yaml
from tqdm import tqdm

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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
        default=8,
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
        '--part',
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
        default='ntu',
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
    parser.add_argument(
        '--debug',
        type=str2bool,
        default=False,
        help='Debug mode; default false')
    parser.add_argument(
        '--assume-yes',
        action='store_true',
        help='Say yes to every prompt')
    parser.add_argument(
        '--forward-batch-size',
        type=int,
        default=1,
        help='Batch size during forward pass, must be factor of --batch-size')
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='path of previously saved training checkpoint')

    parser.add_argument('--only_train_part', default=False)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)
    return parser

class Processor():
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            # Added control through the command line
            arg.train_feeder_args['debug'] = arg.train_feeder_args['debug'] or self.arg.debug
            logdir = os.path.join(arg.work_dir, 'trainlogs')
            if not arg.train_feeder_args['debug']:
                # logdir = arg.model_saved_name
                if os.path.isdir(logdir):
                    print(f'log_dir {logdir} already exists')
                    if arg.assume_yes:
                        answer = 'y'
                    else:
                        answer = input('delete it? [y]/n:')
                    if answer.lower() in ('y', ''):
                        shutil.rmtree(logdir)
                        print('Dir removed:', logdir)
                    else:
                        print('Dir not removed:', logdir)

                self.train_writer = SummaryWriter(os.path.join(logdir, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(logdir, 'val'), 'val')
            else:
                self.train_writer = SummaryWriter(os.path.join(logdir, 'debug'), 'debug')
        self.load_model()
        self.load_optimizer()
        self.load_lr_scheduler()
        self.load_data()

        self.global_step = 0
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

        # if self.arg.half:
        #     self.print_log('*************************************')
        #     self.print_log('*** Using Half Precision Training ***')
        #     self.print_log('*************************************')
        #     self.model, self.optimizer = apex.amp.initialize(
        #         self.model,
        #         self.optimizer,
        #         opt_level=f'O{self.arg.amp_opt_level}'
        #     )
        #     if self.arg.amp_opt_level != 1:
        #         self.print_log('[WARN] nn.DataParallel is not yet supported by amp_opt_level != "O1"')

        # if type(self.arg.device) is list:
        #     if len(self.arg.device) > 1:
        #         self.print_log(f'{len(self.arg.device)} GPUs available, using DataParallel')
        #         self.model = nn.DataParallel(
        #             self.model,
        #             device_ids=self.arg.device,
        #             output_device=self.output_device
        #         )

    def start(self):
        if self.arg.phase == 'train':
            self.print_log(f'Parameters:\n{pprint.pformat(vars(self.arg))}\n')
            self.print_log(f'Model total number of params: {count_params(self.model)}')
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = ((epoch + 1) % self.arg.save_interval == 0) or (epoch + 1 == self.arg.num_epoch)
                self.train(epoch, save_model=save_model)
                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])

            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Forward Batch Size: {self.arg.forward_batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')

        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args['debug']:
                wf = os.path.join(self.arg.work_dir, 'wrong-samples.txt')
                rf = os.path.join(self.arg.work_dir, 'right-samples.txt')
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')

            self.print_log(f'Model:   {self.arg.model}')
            self.print_log(f'Weights: {self.arg.weights}')

            self.eval(
                epoch=0,
                save_score=self.arg.save_score,
                loader_name=['test'],
                wrong_file=wf,
                result_file=rf
            )

            self.print_log('Done.\n')

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        # self.model = Model(**self.arg.model_args).cuda(output_device)
        self.model = Model(num_class=self.arg.model_args['num_class'], num_joint=self.arg.model_args['num_point'],
                           seg = self.arg.model_args['seg'], args=self.arg,
                           graph=self.arg.model_args['graph'], graph_args=self.arg.model_args['graph_args']).cuda(output_device)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)

        if self.arg.weights:
            # self.global_step = int(self.arg.weights[:-3].split('-')[-1])
            self.global_step = 111
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pkl.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            for w in self.arg.ignore_weights:
                if weights.pop(w, None) is not None:
                    self.print_log('Sucessfully Remove Weights: {}.'.format(w))
                else:
                    self.print_log('Can Not Remove Weights: {}.'.format(w))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(self.model, device_ids=self.arg.device, output_device=self.output_device)
                # self.model = nn.DataParallel(
                #     self.model,
                #     device_ids=self.arg.device,
                #     output_device=output_device)

    def load_lr_scheduler(self):
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=self.arg.step, gamma=0.1)
        if self.arg.checkpoint is not None:
            scheduler_states = torch.load(self.arg.checkpoint)['lr_scheduler_states']
            self.print_log(f'Loading LR scheduler states from: {self.arg.checkpoint}')
            self.lr_scheduler.load_state_dict(scheduler_states)
            self.print_log(f'Starting last epoch: {scheduler_states["last_epoch"]}')
            self.print_log(f'Loaded milestones: {scheduler_states["last_epoch"]}')

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()

        def worker_seed_fn(worker_id):
            # give workers different seeds
            return init_seed(self.arg.seed + worker_id + 1)

        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(

                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=worker_seed_fn)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=worker_seed_fn)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':

            params_dict = dict(self.model.named_parameters())
            params = []

            for key, value in params_dict.items():
                decay_mult = 0.0 if 'bias' in key else 1.0

                lr_mult = 1.0
                weight_decay = 1e-4
                if 'Linear_weight' in key:
                    weight_decay = 1e-3
                elif 'Mask' in key:
                    weight_decay = 0.0

                params += [{'params': value, 'lr': self.arg.base_lr, 'lr_mult': lr_mult, 'decay_mult': decay_mult,
                            'weight_decay': weight_decay}]

            self.optimizer = optim.SGD(
                params,
                momentum=0.9,
                nesterov=self.arg.nesterov)

        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1,
                                              patience=10, verbose=True,
                                              threshold=1e-4, threshold_mode='rel',
                                              cooldown=0)

    def save_weights(self, epoch, out_folder='weights'):
        state_dict = self.model.state_dict()
        weights = OrderedDict([
            [k.split('module.')[-1], v.cpu()]
            for k, v in state_dict.items()
        ])

        weights_name = f'weights-{epoch}-{int(self.global_step)}.pt'
        self.save_states(epoch, weights, out_folder, weights_name)

    def train(self, epoch, save_model=False):
        self.model.train()
        loader = self.data_loader['train']
        loss_values = []
        self.train_writer.add_scalar('epoch', epoch + 1, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        current_lr = self.optimizer.param_groups[0]['lr']
        self.print_log(f'Training epoch: {epoch + 1}, LR: {current_lr:.4f}')

        process = tqdm(loader, dynamic_ncols=True)
        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            # get data
            data = Variable(data.float().cuda(self.output_device), requires_grad=False)
            label = Variable(label.long().cuda(self.output_device), requires_grad=False)
            timer['dataloader'] += self.split_time()

            self.model.zero_grad()
            self.optimizer.zero_grad()
            start = time.time()
            ############## Gradient Accumulation for Smaller Batches ##############
            real_batch_size = self.arg.forward_batch_size
            splits = len(data) // real_batch_size
            assert len(data) % real_batch_size == 0, \
                'Real batch size should be a factor of arg.batch_size!'

            for i in range(splits):
                left = i * real_batch_size
                right = left + real_batch_size
                batch_data, batch_label = data[left:right], label[left:right]

                # forward
                output = self.model(batch_data)
                if isinstance(output, tuple):
                    output, l1 = output
                    l1 = l1.mean()
                else:
                    l1 = 0

                loss = self.loss(output, batch_label) / splits

                loss.backward()

                # if self.arg.half:
                #     with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                #         scaled_loss.backward()
                # else:
                #     loss.backward()

                loss_values.append(loss.item())
                timer['model'] += self.split_time()

                # Display loss
                process.set_description(f'(Setting BS:{len(data)}, Real BS {real_batch_size}) loss: {loss.item():.4f}')

                value, predict_label = torch.max(output, 1)
                acc = torch.mean((predict_label == batch_label).float())
                # self.train_writer.add_scalar('acc', acc, self.global_step)
                # self.train_writer.add_scalar('loss', loss.item() * splits, self.global_step)
                # self.train_writer.add_scalar('loss_l1', l1, self.global_step)
            #####################################

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
            self.optimizer.step()
            network_time = time.time() - start
            # # forward
            # start = time.time()
            # output = self.model(data)
            # network_time = time.time() - start
            #
            # loss = self.loss(output, label)
            #
            # # backward
            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()
            # loss_values.append(loss.data)
            # timer['model'] += self.split_time()
            #
            # value, predict_label = torch.max(output.data, 1)
            # acc = torch.mean((predict_label == label.data).float())

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']

            if self.global_step % self.arg.log_interval == 0:
                self.print_log(
                    '\tBatch({}/{}) done. Acc: {:.6f} Loss: {:.4f}  lr:{:.6f}  network_time: {:.4f}'.format(
                        batch_idx, len(loader), acc, loss.data, self.lr, network_time))
            timer['statistics'] += self.split_time()

            # Delete output/loss after each batch since it may introduce extra mem during scoping
            # https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770/3
            del output
            del loss

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }

        mean_loss = np.mean(loss_values)
        num_splits = self.arg.batch_size // self.arg.forward_batch_size
        self.print_log(
            f'\tMean training loss: {mean_loss:.4f} (BS {self.arg.batch_size}: {mean_loss * num_splits:.4f}).')
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        # PyTorch > 1.2.0: update LR scheduler here with `.step()`
        # and make sure to save the `lr_scheduler.state_dict()` as part of checkpoint
        self.lr_scheduler.step()

        if save_model:
            # save training checkpoint & weights
            self.save_weights(epoch + 1)
            self.save_checkpoint(epoch + 1)

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        # Skip evaluation if too early
        self.arg.eval_start = 1
        if epoch + 1 < self.arg.eval_start:
            return

        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        with torch.no_grad():
            self.model = self.model.cuda(self.output_device)
            self.model.eval()
            self.print_log(f'Eval epoch: {epoch + 1}')
            for ln in loader_name:
                loss_values = []
                score_batches = []
                step = 0
                process = tqdm(self.data_loader[ln], dynamic_ncols=True)
                for batch_idx, (data, label, index) in enumerate(process):
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    output = self.model(data)
                    if isinstance(output, tuple):
                        output, l1 = output
                        l1 = l1.mean()
                    else:
                        l1 = 0
                    loss = self.loss(output, label)
                    score_batches.append(output.data.cpu().numpy())
                    loss_values.append(loss.item())

                    _, predict_label = torch.max(output.data, 1)
                    step += 1

                    if wrong_file is not None or result_file is not None:
                        predict = list(predict_label.cpu().numpy())
                        true = list(label.data.cpu().numpy())
                        for i, x in enumerate(predict):
                            if result_file is not None:
                                f_r.write(str(x) + ',' + str(true[i]) + '\n')
                            if x != true[i] and wrong_file is not None:
                                f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')

            score = np.concatenate(score_batches)
            loss = np.mean(loss_values)
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1

            print('Accuracy: ', accuracy, ' model: ', self.arg.work_dir)
            if self.arg.phase == 'train' and not self.arg.debug:
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('loss_l1', l1, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log(f'\tMean {ln} loss of {len(self.data_loader[ln])} batches: {np.mean(loss_values)}.')
            for k in self.arg.show_topk:
                self.print_log(f'\tTop {k}: {100 * self.data_loader[ln].dataset.top_k(score, k):.2f}%')

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pkl.dump(score_dict, f)

        # Empty cache after evaluation
        torch.cuda.empty_cache()

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

    def save_states(self, epoch, states, out_folder, out_name):
        out_folder_path = os.path.join(self.arg.work_dir, out_folder)
        out_path = os.path.join(out_folder_path, out_name)
        os.makedirs(out_folder_path, exist_ok=True)
        torch.save(states, out_path)

    def save_checkpoint(self, epoch, out_folder='checkpoints'):
        state_dict = {
            'epoch': epoch,
            'optimizer_states': self.optimizer.state_dict(),
            'lr_scheduler_states': self.lr_scheduler.state_dict(),
        }

        checkpoint_name = f'checkpoint-{epoch}-fwbz{self.arg.forward_batch_size}-{int(self.global_step)}.pt'
        self.save_states(epoch, state_dict, out_folder, checkpoint_name)

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
        with open(os.path.join(self.arg.work_dir, self.arg.part + '_config.yaml'), 'w') as f:
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
    
