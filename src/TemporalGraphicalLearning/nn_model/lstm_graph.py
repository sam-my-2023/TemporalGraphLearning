from typing import Any
import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,Sequential

from torch_geometric.utils import spmm,dense_to_sparse,to_dense_adj
import numpy as np


from torch.nn import MultiheadAttention

class Eval_Dataset:
    def __init__(self,x,pre_l,step_days = 1, dropout=0):
        self.sizes = {key:_.size() for key,_ in x.items()}
        self.keys = list(self.sizes.keys())
        # seq_len = 1e5
        # for key,x_ in x.items():
        #     if x_.size(0)>1:
        #         seq_len = min(x_.size(0),seq_len)

        self.x = x
        
        self.step_days = step_days
        self.pre_l = pre_l
        
        self.n = 1e5
        self.m = 1e5
        self.n = min(self.n,x['data'].size(0) -pre_l)
        self.n = min(self.n,x['graph'].size(0)-pre_l+1)
        self.n = min(self.n,x['tag'].size(0))
        
        self.m = min(self.m,x['data'].size(1), x['tag'].size(1),x['graph'].size(1))
        self.sample_n = int(self.m*dropout)
    
    def __getitem__(self,i):
        return tuple(self.getdata(key,i) for key in self.keys)
    
    def getdata(self, name: str, i):
        if self.sample_n >0:
            idx = np.random.choice(self.m,self.sample_n, replace=False)
            return self.sub_data(idx,name,i)
        
        if name == 'data':
            return self.x['data'][i:i+self.pre_l:self.step_days]
        elif name == 'tag':
            return self.x['tag'][i+self.pre_l-1]
        elif name == 'graph':
            return torch.mean(self.x['graph'][i:i+self.pre_l:self.step_days],dim=0)
        else:
            raise
        
    def sub_data(self,idx,name,i):
        if name == 'data':
            return self.x['data'][i:i+self.pre_l:self.step_days,idx,:]
        elif name == 'tag':
            return self.x['tag'][i+self.pre_l-1,idx]
        elif name == 'graph':
            return torch.mean(self.x['graph'][i:i+self.pre_l:self.step_days],dim=0)[idx,:][:,idx]
        else:
            raise
    
    def __len__(self):
        return self.n
    
    def size(self, kth_dim=None):
        size = (self.n,self.m)
        if kth_dim is None:
            return size
        else:
            return size[kth_dim]
        
class LSTM_only(nn.Module):
    def __init__(self, rnn_layer, out_layer):
        super().__init__()
        self.rnn_layer = rnn_layer
        self.out_layer = out_layer
    
    def forward(self, batch_inputs, batch_graph):
        '''
        the input dims follows (batch, seq_length, num_companies, feature_size) 
        However, for the convenience (without set batch_first=True in LSTM).
        the input dim is permuted as -> (seq_length, batch_size, num_companies, feature_size ) if data is batched.
        '''
        
        if batch_inputs.dim() == 4:
            batch_inputs = batch_inputs.permute(1,0,2,3)
            assert batch_inputs.size(1) == batch_graph.size(0) and batch_graph.dim() == 3
            batch_inputs = batch_inputs.flatten(start_dim= 1,end_dim=2)
        seq_embedding, (hidden_rnn_embedding,_)= self.rnn_layer(batch_inputs)
        rnn_embedding = hidden_rnn_embedding.permute(1,0,2)
        rnn_embedding = rnn_embedding.reshape(rnn_embedding.size(0), -1)
        # aggregate_graphs_along_seq_dim = torch.sum(batch_graph,dim=0)
        output = self.out_layer(rnn_embedding)
        # output = output.unflatten(dim=0, sizes= flatten_size)
        return output
    

class LSTM_GCNN(nn.Module):
    def __init__(self, graph_layer,mid_layer,rnn_layer, out_layer=None):
        super().__init__()
        self.rnn_layer = rnn_layer
        self.mid_layer = mid_layer
        self.gcnn_layer = graph_layer
    
    def forward(self, batch_inputs, batch_graph):
        if batch_inputs.dim() == 4:
            batch_inputs = batch_inputs.permute(1,0,2,3)
            assert batch_inputs.size(1) == batch_graph.size(0) and batch_graph.dim() == 3
            batch_inputs = batch_inputs.flatten(start_dim= 1,end_dim=2)
        seq_embedding, (hidden_rnn_embedding,_)= self.rnn_layer(batch_inputs)
        rnn_embedding = hidden_rnn_embedding.permute(1,0,2)
        rnn_embedding = rnn_embedding.reshape(rnn_embedding.size(0), -1)
        rnn_embedding = self.mid_layer(rnn_embedding)
        # aggregate_graphs_along_seq_dim = torch.sum(batch_graph,dim=0)
        edge_index,_ = dense_to_sparse(batch_graph)
        output = self.gcnn_layer(rnn_embedding, edge_index)
        output = output.unsqueeze(0)
        # output = output.unflatten(dim=0, sizes= flatten_size)
        return seq_embedding, output
    

@torch.no_grad()
def eval(our_model,test_ds):
    pred = []
    ground_truth = []
    for seq_stock,tag,seq_graph in test_ds:
        logits = our_model(seq_stock,seq_graph)
        tem = torch.argmax(logits, dim=1)
        pred.append(tem.to('cpu').detach().numpy())
        ground_truth.append(tag.to('cpu').detach().numpy())
    pred = np.concatenate(pred)
    ground_truth = np.concatenate(ground_truth)
    return pred,ground_truth

