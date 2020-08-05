# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from torch import nn
import torch
import math
import numpy as np

import sys
#sys.path.append("./Temporal_shift/")

#from cuda.shift import Shift

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def conv_init(conv):
    nn.init.kaiming_normal(conv.weight, mode='fan_out')
    nn.init.constant(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant(bn.weight, scale)
    nn.init.constant(bn.bias, 0)

class tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

# class Shift_tcn(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, bias=True, tem=None, seg=1):
#         super(Shift_tcn, self).__init__()
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#
#         self.bn = nn.BatchNorm2d(in_channels)
#         self.bn2 = nn.BatchNorm2d(in_channels)
#         bn_init(self.bn2, 1)
#         self.relu = nn.ReLU(inplace=True)
#         self.shift_in = Shift(channel=in_channels, stride=1, init_scale=1)
#         self.shift_out = Shift(channel=out_channels, stride=stride, init_scale=1)
#
#         self.temporal_linear = nn.Conv2d(in_channels, out_channels, 1)
#         nn.init.kaiming_normal(self.temporal_linear.weight, mode='fan_out')
#
#         # Start: Integrate
#         self.tem = tem
#         self.seg = seg
#         self.tem_embed = embed(self.seg, 64 * 4, norm=False, bias=bias)
#         # End: Integrate
#
#     def forward(self, x):
#         # TODO
#         # #Start: Integrate
#         # tem1 = self.tem_embed(self.tem)
#         # x = x + tem1
#         # x = self.cnn(x)
#         # #End: Integrate
#
#         x = self.bn(x)
#         # shift1
#         x = self.shift_in(x)
#         x = self.temporal_linear(x)
#         x = self.relu(x)
#         # shift2
#         x = self.shift_out(x)
#         x = self.bn2(x)
#         return x


class Shift_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, metric, A, coff_embedding=4, num_subset=3, bias=True):
        #def __init__(self, in_channels, out_channels, metric, A, coff_embedding=4, num_subset=3, bias=True):
        super(Shift_gcn, self).__init__()
        # Shift model part

        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        # NTU data metric shift number
        if metric == 'upper':
            self.shift_size = 19
        else:
            self.shift_size = 13

        # self.bn = nn.BatchNorm1d(self.shift_size*out_channels)
        # self.bn_semantic = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU()

        self.Linear_weight = nn.Parameter(torch.zeros(in_channels, out_channels, requires_grad=True, device='cuda'),
                                          requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0, math.sqrt(1.0 / out_channels))

        self.Linear_bias = nn.Parameter(torch.zeros(1, 1, out_channels, requires_grad=True, device='cuda'),
                                        requires_grad=True)
        nn.init.constant(self.Linear_bias, 0)

        self.Feature_Mask = nn.Parameter(torch.ones(1, self.shift_size, in_channels, requires_grad=True, device='cuda'),
                                         requires_grad=True)
        nn.init.constant(self.Feature_Mask, 0)

        self.bn = nn.BatchNorm1d(self.shift_size * out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

        # 19 joints in upper and middle partition
        index_array = np.empty(self.shift_size * in_channels).astype(np.int)
        for i in range(self.shift_size):
            for j in range(in_channels):
                index_array[i * in_channels + j] = (i * in_channels + j + j * in_channels) % (
                            in_channels * self.shift_size)
        self.shift_in = nn.Parameter(torch.from_numpy(index_array), requires_grad=False)

        index_array = np.empty(self.shift_size * out_channels).astype(np.int)
        for i in range(self.shift_size):
            for j in range(out_channels):
                index_array[i * out_channels + j] = (i * out_channels + j - j * out_channels) % (
                            out_channels * self.shift_size)
        self.shift_out = nn.Parameter(torch.from_numpy(index_array), requires_grad=False)

        # Semantic model part
        self.dim1 = out_channels * 2
        num_class = 60
        #self.dataset = dataset
        #self.seg = seg

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.cnn = local(self.dim1, self.dim1 * 2, bias=bias)
        self.compute_g1 = compute_g_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.sgcn1 = sgcn_spa(self.dim1 // 2, self.dim1 // 2, bias=bias)
        self.sgcn2 = sgcn_spa(self.dim1 // 2, self.dim1, bias=bias)
        #self.sgcn3 = sgcn_spa(self.dim1, self.dim1, bias=bias)
        self.sgcn3 = sgcn_spa(self.dim1, out_channels, bias=bias)
        self.fc = nn.Linear(self.dim1 * 2, num_class)
        #self.fc = nn.Linear(self.dim1, out_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        nn.init.constant_(self.sgcn1.w.cnn.weight, 0)
        nn.init.constant_(self.sgcn2.w.cnn.weight, 0)
        nn.init.constant_(self.sgcn3.w.cnn.weight, 0)

        # self.w = cnn1x1(self.out_channels, self.out_channels, bias=False)
        # self.w1 = cnn1x1(self.out_channels, self.out_channels, bias=bias)
        #
        # self.compute_g1 = compute_g_spa(self.out_channels, self.out_channels, bias=bias)

    def forward(self, x0):
        # SEMANTIC
        # x = x1.permute(0, 3, 2, 1).contiguous()
        # x = g.matmul(x)
        # x = x.permute(0, 3, 2, 1).contiguous()
        # x = self.w(x) + self.w1(x1)
        # x = self.relu(self.bn(x))

        # SHIFT
        # n, c, t, v = x1.size()
        n, c, t, v = x0.size()
        x = x0.permute(0, 2, 3, 1).contiguous()

        # shift1
        x = x.view(n * t, v * c)
        x = torch.index_select(x, 1, self.shift_in)
        x = x.view(n * t, v, c)
        x = x * (torch.tanh(self.Feature_Mask) + 1)

        x = torch.einsum('nwc,cd->nwd', (x, self.Linear_weight)).contiguous()  # nt,v,c
        x = x + self.Linear_bias

        # shift2
        x = x.view(n * t, -1)
        x = torch.index_select(x, 1, self.shift_out)
        x = self.bn(x)
        x = x.view(n, t, v, self.out_channels).permute(0, 3, 1, 2)  # n,c,t,v

        x = x + self.down(x0)
        x = self.relu(x)

        # Sematinc SGN
        # x1 = x1.permute(0, 1, 3, 2).contiguous()
        # x = x1.permute(0, 3, 2, 1).contiguous()
        # x = g.matmul(x)
        # x = x.permute(0, 3, 2, 1).contiguous()
        # x = self.w(x) + self.w1(x1)
        # x = self.relu(self.bn_semantic(x))

        g = self.compute_g1(x)
        x = self.sgcn1(x, g)
        x = self.sgcn2(x, g)
        x = self.sgcn3(x, g)

        #x = self.fc(x)

        # x = x.permute(0, 1, 3, 2).contiguous()
        # x1 = x
        # x = x.permute(0, 3, 2, 1).contiguous()
        # x = g.matmul(x)
        # x = x.permute(0, 3, 2, 1).contiguous()
        # x = self.w(x) + self.w1(x1)
        # x = self.relu(self.bn_semantic(x))
        #
        # x = x.permute(0, 1, 3, 2)

        return x


class TCN_GCN_unit(nn.Module):
    # def __init__(self, in_channels, out_channels, metric, A, stride=1, residual=True, tem=None, seg=1):
    def __init__(self, in_channels, out_channels, metric, A, stride=1, residual=False, tem=None):
        super(TCN_GCN_unit, self).__init__()

        self.gcn1 = Shift_gcn(in_channels, out_channels, metric, A)
        #self.tcn1 = Shift_tcn(out_channels, out_channels, stride=stride, tem=tem)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        # x = self.tcn1(self.gcn1(x)) + self.residual(x)
        # return self.relu(x)
        # return x
        x = self.gcn1(x) + self.residual(x)
        return self.relu(x)



class Model(nn.Module):
    #def __init__(self, num_class, seg, args, bias=True, graph=None, graph_args=dict()):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3, arg=None, metric=None):
        super(Model, self).__init__()

        self.metric = metric
        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.l1 = TCN_GCN_unit(3, 64, self.metric, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, self.metric, A)
        self.l3 = TCN_GCN_unit(64, 64, self.metric, A)
        self.l4 = TCN_GCN_unit(64, 64, self.metric, A)
        self.l5 = TCN_GCN_unit(64, 128, self.metric, A, stride=2)
        # self.l6 = TCN_GCN_unit(128, 128, self.metric, A)
        # self.l7 = TCN_GCN_unit(128, 128, self.metric, A)
        self.l8 = TCN_GCN_unit(128, 256, self.metric, A, stride=2)
        # self.l9 = TCN_GCN_unit(256, 256, self.metric, A)
        # self.l10 = TCN_GCN_unit(256, 256, self.metric, A)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        # x = self.l6(x)
        # x = self.l7(x)
        x = self.l8(x)
        # x = self.l9(x)
        # x = self.l10(x)

        c_new = x.size(1)
        x = x.reshape(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        return self.fc(x)

        #return output
    def one_hot(self, bs, spa, tem):
        # spa: number of joints
        y = torch.arange(spa).unsqueeze(-1)
        y_onehot = torch.FloatTensor(spa, spa)

        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)

        y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)
        y_onehot = y_onehot.repeat(bs, tem, 1, 1)

        return y_onehot

class norm_data(nn.Module):
    def __init__(self, dim= 64):
        super(norm_data, self).__init__()

        #self.bn = nn.BatchNorm1d(dim* 25)
        self.bn = nn.BatchNorm1d(dim * 18)

    def forward(self, x):
        bs, c, num_joints, step = x.size()
        x = x.view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, -1, num_joints, step).contiguous()
        return x

class embed(nn.Module):
    # Original
    #def __init__(self, dim=3, dim1=128, norm=True, bias=False):
    # Hao:
    def __init__(self, dim = 2, dim1 = 128, norm = True, bias = False):
        super(embed, self).__init__()

        if norm:
            self.cnn = nn.Sequential(
                norm_data(dim),
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )

    def forward(self, x):
        x = self.cnn(x)
        return x

class cnn1x1(nn.Module):
    # Original:
    # def __init__(self, dim1 = 3, dim2 =3, bias = True):
    # Hao:
    def __init__(self, dim1 = 3, dim2 =3, bias = True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x

class local(nn.Module):
    def __init__(self, dim1 = 3, dim2 = 3, bias = False):
        super(local, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((1, 20))
        self.cnn1 = nn.Conv2d(dim1, dim1, kernel_size=(1, 3), padding=(0, 1), bias=bias)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU()
        self.cnn2 = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim2)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x1):
        x1 = self.maxpool(x1)
        x = self.cnn1(x1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class sgcn_spa(nn.Module):
    def __init__(self, in_feature, out_feature, bias = False):
        super(sgcn_spa, self).__init__()
        self.bn = nn.BatchNorm2d(out_feature)
        self.relu = nn.ReLU()
        self.w = cnn1x1(in_feature, out_feature, bias=False)
        self.w1 = cnn1x1(in_feature, out_feature, bias=bias)

    def forward(self, x1, g):
        # x = x1.permute(0, 3, 2, 1).contiguous()
        x = x1.permute(0, 2, 3, 1).contiguous()
        x = g.matmul(x)
        #x = x.permute(0, 3, 2, 1).contiguous()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.w(x) + self.w1(x1)
        x = self.relu(self.bn(x))
        return x

class compute_g_spa(nn.Module):
    def __init__(self, dim1 = 64 *3, dim2 = 64*3, bias = False):
        super(compute_g_spa, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.g1 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.g2 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1):
        g1 = self.g1(x1).permute(0, 2, 3, 1).contiguous()
        g2 = self.g2(x1).permute(0, 2, 1, 3).contiguous()
        g3 = g1.matmul(g2)
        g = self.softmax(g3)
        return g
    
