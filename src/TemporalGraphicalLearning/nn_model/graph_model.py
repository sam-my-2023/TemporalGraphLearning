import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,Sequential

from torch_geometric.utils import spmm,dense_to_sparse,to_dense_adj
import numpy as np

class Eval_Dataset:
    def __init__(self,data_per_date_per_company,tag_per_date_per_company, adj_per_date):
        self.n = min(adj_per_date.shape[0],                    
                     data_per_date_per_company.shape[0],
                     tag_per_date_per_company.shape[0])
        
        self.num_companies = data_per_date_per_company.shape[1]
        assert self.num_companies == tag_per_date_per_company.shape[1]
        self.adj_per_date = adj_per_date[:self.n]
        self.data_per_date_per_company = data_per_date_per_company
        self.tag_per_date_per_company = tag_per_date_per_company
    
    def __getitem__(self,i):
        return self.data_per_date_per_company[i],self.tag_per_date_per_company[i],self.adj_per_date[i]
    
    def __len__(self):
        return self.n
    

class GCNN(nn.Module):
    def __init__(self, graph_layer):
        super().__init__()
        self.gcnn_layer = graph_layer
    
    def forward(self, batch_inputs, batch_graph):
        assert batch_inputs.dim() == batch_graph.dim()
        if batch_inputs.dim() == 3:
            flatten_size = batch_inputs.size()[:2]
            batch_inputs = batch_inputs.flatten(end_dim=1)
        else:
            flatten_size = batch_inputs.size()[:1]
        batch_graph,_ = dense_to_sparse(batch_graph)
        output = self.gcnn_layer(batch_inputs,batch_graph)
        # output = output.unflatten(dim=0, sizes= flatten_size)
        return output

# class Eval_Dataset:
#     def __init__(self,data_per_date_per_company,tag_per_date_per_company, adj_per_date, r,l):
#         self.r = r
#         self.l = l
#         self.n = min(adj_per_date.shape[0],                    
#                      data_per_date_per_company.shape[0]-r,
#                      tag_per_date_per_company.shape[0]-2*r-l)
        
#         self.num_companies = data_per_date_per_company.shape[1]
#         assert self.num_companies == tag_per_date_per_company.shape[1]
#         self.adj_per_date = adj_per_date[:self.n]
#         self.data_per_date_per_company = data_per_date_per_company[:self.n + r]
#         self.tag_per_date_per_company = tag_per_date_per_company[:self.n + 2*r+l]
    
#     def __getitem__(self,i):
#         return self.data_per_date_per_company[i+self.r],self.tag_per_date_per_company[i+2*self.r+self.l],self.adj_per_date[i]
    
#     def __len__(self):
#         return self.n

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
    
        
