import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,Sequential

from torch_geometric.utils import spmm,dense_to_sparse,to_dense_adj
import numpy as np

class Eval_Dataset:
    def __init__(self,data_per_date_per_company,tag_per_date_per_company, adj_per_date,pre_l):
        self.n = min(adj_per_date.shape[0]- pre_l+1,                    
                     data_per_date_per_company.shape[0] - pre_l+1,
                     tag_per_date_per_company.shape[0])
        
        self.pre_l = pre_l
        
        self.num_companies = data_per_date_per_company.shape[1]
        assert self.num_companies == tag_per_date_per_company.shape[1]
        self.adj_per_date = adj_per_date
        self.data_per_date_per_company = data_per_date_per_company
        self.tag_per_date_per_company = tag_per_date_per_company
    
    def __getitem__(self,i):
        return self.data_per_date_per_company[i:i+self.pre_l],self.tag_per_date_per_company[i], self.adj_per_date[i:i+self.pre_l]
    
    def __len__(self):
        return self.n
    

class GCNN_RNN(nn.Module):
    def __init__(self, graph_layer,rnn_layer, out_layer):
        super().__init__()
        self.rnn_layer = rnn_layer
        self.gcnn_layer = graph_layer
        self.out_layer = out_layer
    
    def forward(self, batch_inputs, batch_graph):
        '''
        the input dims follows (batch, seq_length, num_companies, feature_size) 
        '''
        
        if batch_inputs.dim() == 4:
            assert batch_graph.dim() == 4
            batch_inputs = batch_inputs.permute(1,0,2,3)
        
        seq_size = batch_inputs.size(0)
        batch_inputs = batch_inputs.flatten(end_dim=-2)
        batch_graph = batch_graph.flatten(end_dim=-3)

        edge_index,_ = dense_to_sparse(batch_graph)
        graph_embedding = self.gcnn_layer(batch_inputs, edge_index)
        graph_embedding = graph_embedding.reshape(seq_size,-1,graph_embedding.size(-1))



        seq_embedding, (hidden_rnn_embedding,_)= self.rnn_layer(graph_embedding)
        rnn_embedding = hidden_rnn_embedding.permute(1,0,2)
        rnn_embedding = rnn_embedding.reshape(rnn_embedding.size(0), -1)
        # aggregate_graphs_along_seq_dim = torch.sum(batch_graph,dim=0)
        
        output = self.out_layer(rnn_embedding)
        # output = output.unflatten(dim=0, sizes= flatten_size)
        return output


@torch.no_grad()
def eval(our_model,test_ds, device = 'cpu'):
    pred = []
    ground_truth = []
    for seq_stock,tag,seq_graph in test_ds:
        logits = our_model(seq_stock,seq_graph)
        tem = torch.argmax(logits, dim=1)
        pred.append(tem.to(device).detach().numpy())
        ground_truth.append(tag.to(device).detach().numpy())
    pred = np.concatenate(pred)
    ground_truth = np.concatenate(ground_truth)
    return pred,ground_truth