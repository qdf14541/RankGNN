import os
import numpy as np
import torch
import dgl
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')

import copy
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from gumble_vector_quantizer import GumbelVectorQuantizer

class QuantLayer(nn.Module):
    
    def __init__(self, quant_combine, in_dim, out_dim, num_vars=8, groups=8):
        super(QuantLayer, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.quantizer = GumbelVectorQuantizer(
            dim=32,
            num_vars=num_vars,
            temp=(2, 0.5, 0.999995),
            groups=groups,
            combine_groups=quant_combine,
            vq_dim=groups*64,
            weight_proj_depth=1,
            weight_proj_factor=3,
        )
        self.quantizer_preproject = nn.Linear(in_dim, 32)
        self.quantizer_postproject = nn.Linear(groups*64, out_dim)
        
    def set_num_updates(self, num_updates):
        self.quantizer.set_num_updates(num_updates)
        
    def forward(self, x):
        x = self.quantizer_preproject(x)
        
        res = self.quantizer(x)
        x = res["x"]
        
        x = self.quantizer_postproject(x)
        
        return x
    
class RelationLayer(nn.Module):
    def __init__(self, input_dim, output_dim, head, etype, dropout, if_sum=False):
        super().__init__()
        self.etype = etype
        self.head = head
        self.hd = output_dim
        self.if_sum = if_sum
        
        self.atten = nn.Linear(2*self.hd, 1)
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.q_base = nn.Linear(input_dim, output_dim)
        self.q_bias = nn.Linear(input_dim, output_dim)
        self.bias = nn.Linear(output_dim, output_dim, bias=False)
        
        self.base = nn.Linear(output_dim, output_dim, bias=False)
        
        self.gate = nn.Linear(output_dim*2, output_dim*head)
        
        self.w_linear = nn.Linear(input_dim, output_dim*head)

    def forward(self, g, h):
        with g.local_scope():

            g.ndata['h'] = self.w_linear(h)
            
            g.update_all(message_func=self.message, reduce_func=self.reduce, etype=self.etype)
            
            out = g.ndata['out']
            return out

    def message(self, edges):
        src = edges.src
        src_features = src['h']
        src_features = src_features.view(-1, self.head, self.hd)
        
        z = torch.cat([src_features, edges.dst['h'].view(-1, self.head, self.hd)], dim=-1)
        
        alpha = self.atten(z)
        alpha = self.leakyrelu(alpha)
        
        return {'atten':alpha, 'sf':src_features}

    def reduce(self, nodes):
        alpha = nodes.mailbox['atten']
        sf = nodes.mailbox['sf']
        
        alpha = self.softmax(alpha)
        out = torch.sum(alpha*sf, dim=1)
        if not self.if_sum:
            out = out.view(-1, self.head*self.hd)
        else:
            out = out.sum(dim=-2)
            
        return {'out':out}
    
class MultiRelationLayer(nn.Module):
    def __init__(self, input_dim, output_dim, head, dataset, dropout, if_sum=False, quant_combine=True, use_quant=False, num_vars=8, groups=16):
        super(MultiRelationLayer, self).__init__()
        
        self.use_quant = use_quant
        self.relation = copy.deepcopy(dataset.etypes)
        self.relation.remove('homo')
        self.n_relation = len(self.relation)
        qdim = self.n_relation*output_dim*head
        if not if_sum:
            self.linear = nn.Linear(self.n_relation*output_dim*head, output_dim*head)
            self.mlp = nn.Linear(input_dim, 1)
            self.mlp2 = nn.Linear(input_dim, self.n_relation*output_dim*head)
        else:
            qdim = self.n_relation*output_dim
            self.linear = nn.Linear(self.n_relation*output_dim, output_dim)
            self.mlp = nn.Linear(input_dim, 1)
            self.mlp2 = nn.Linear(input_dim, self.n_relation*output_dim)
            
        self.minelayers = nn.ModuleDict()
        self.dropout = nn.Dropout(dropout)

        for e in self.relation:
            self.minelayers[e] = RelationLayer(input_dim, output_dim, head, e, dropout, if_sum)
        
        if self.use_quant:
            self.quantizer = QuantLayer(quant_combine, qdim, qdim, num_vars=num_vars, groups=groups)
        
    def set_num_updates(self, num_update):
        if self.use_quant:
            self.quantizer.set_num_updates(num_update)

    def forward(self, g, h):
        h2 = self.mlp(h)
        h3 = self.mlp2(h)
        
        hs = []
        for e in self.relation:
            he = self.minelayers[e](g, h)
            hs.append(he)
        h = torch.cat(hs, dim=1)
        h = self.dropout(h)
        
        if self.use_quant:
            h = self.quantizer(h)
        
        h = h + F.leaky_relu(h3)
        
        h1 = self.linear(h)
        
        return h1, h2
    
    def loss(self, g, h, ratio):
        with g.local_scope():
            # MASK= g.adj().todense()
            MASK = g.adj(etype=g.etypes[0]).bool().to_dense()  # will be different connection types
            for i in range(1, len(g.etypes)):
                MASK = torch.logical_or(MASK, g.adj(etype=g.etypes[i]).bool().to_dense())            
            g.ndata['feat'] = h
            agg_h, h2 = self.forward(g, h)

            train_mask = g.ndata['train_mask'].bool()
            train_h = h2[train_mask]
            train_label = g.ndata['label'][train_mask]
            MASK = MASK.to(train_mask.device)
            MASK = MASK[train_mask, :][:, train_mask]

            loss, inter_y, inter_pre, connection = rank_loss(train_label, train_h, ratio=ratio, mask=MASK)

            return agg_h, loss, inter_y, inter_pre, connection
    
    
class Model(nn.Module):
    def __init__(self, args, g):
        super().__init__()
        self.n_layer = args.n_layer
        self.input_dim = g.nodes['r'].data['feature'].shape[1]
        self.intra_dim = args.intra_dim
        self.n_class = args.n_class
        self.n_layer = args.n_layer
        self.use_quant = args.use_quant
        self.only_last = args.only_last
        self.ratio = args.ratio
        self.rank_ratio = args.rank_ratio
        #self.loss_ratio = args.loss_ratio
        self.quant_combine = args.quant_combine
        
        self.mine_layers = nn.ModuleList()
        if args.n_layer == 1:
            self.mine_layers.append(MultiRelationLayer(self.input_dim, self.n_class, args.head, g, args.dropout, if_sum=True, quant_combine=args.quant_combine, use_quant=self.use_quant, num_vars=args.num_vars, groups=args.groups))
        else:
            tmp_quant = self.use_quant
            if self.only_last:
                tmp_quant = False
            self.mine_layers.append(MultiRelationLayer(self.input_dim, self.intra_dim, args.head, g, args.dropout, quant_combine=args.quant_combine, use_quant=tmp_quant, num_vars=args.num_vars, groups=args.groups))
            
            for _ in range(1, self.n_layer-1):
                self.mine_layers.append(MultiRelationLayer(self.intra_dim*args.head, self.intra_dim, args.head, g, args.dropout, quant_combine=args.quant_combine, use_quant=self.use_quant, num_vars=args.num_vars, groups=args.groups))
            
            if self.only_last and self.use_quant:
                tmp_quant = True
                
            self.mine_layers.append(MultiRelationLayer(self.intra_dim*args.head, self.n_class, args.head, g, args.dropout, if_sum=True, quant_combine=args.quant_combine, use_quant=tmp_quant, num_vars=args.num_vars, groups=args.groups))
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()
        
    def set_num_updates(self, num_updates):
        if self.use_quant:
            if self.only_last:
                self.mine_layers[-1].set_num_updates(num_updates)
            else:
                for layer in self.mine_layers:
                    layer.set_num_updates(num_updates)
        
    def forward(self, g):
        feats = g.ndata['feature'].float()
        h, _ = self.mine_layers[0](g, feats)
        if self.n_layer > 1:
            h = self.relu(h)
            h = self.dropout(h)
            for i in range(1, len(self.mine_layers)-1):
                h, _ = self.mine_layers[i](g, h)
                h = self.relu(h)
                h = self.dropout(h)
            h, _ = self.mine_layers[-1](g, h)
        return h 
    
    def loss(self, g):
        feats = g.ndata['feature'].float()
        train_mask = g.ndata['train_mask'].bool()
        train_label = g.ndata['label'][train_mask]
        
        rankloss = 0.0
        
        h, rloss, inter_y, inter_pre, connection = self.mine_layers[0].loss(g, feats, self.ratio)
        rankloss += rloss
        Inter_y = copy.deepcopy(inter_y)
        Inter_pre = copy.deepcopy(inter_pre)
        Connection = copy.deepcopy(connection)
        if self.n_layer > 1:
            h = self.relu(h)
            h = self.dropout(h)
            for i in range(1, len(self.mine_layers)-1):
                h, rloss, inter_y, inter_pre, connection = self.mine_layers[i].loss(g, h, self.ratio)
                h = self.relu(h)
                h = self.dropout(h)
                rankloss += rloss
                Inter_y = torch.cat((Inter_y, inter_y), dim=1)
                Inter_pre = torch.cat((Inter_pre, inter_pre), dim=1)
                Connection = torch.cat((Connection, connection), dim=1)
            h, rloss, inter_y, inter_pre, connection = self.mine_layers[-1].loss(g, h, self.ratio)
            rankloss += rloss
            Inter_y = torch.cat((Inter_y, inter_y), dim=1)  # sample_nums * layers
            Inter_pre = torch.cat((Inter_pre, inter_pre), dim=1)
            Connection = torch.cat((Connection, connection), dim=1)

        # loss = rank_loss(train_label, h[train_mask])
        _, cnt = torch.unique(train_label, return_counts=True)
        loss_fn = torch.nn.CrossEntropyLoss(weight=1 / cnt)
        loss = loss_fn(h[train_mask], train_label.long()) + self.rank_ratio*rankloss/len(self.mine_layers)
        return loss, Inter_y, Inter_pre, Connection

