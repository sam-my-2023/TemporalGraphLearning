
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,Sequential

from torch_geometric.utils import spmm,dense_to_sparse,to_dense_adj
import numpy as np

class Split(nn.Module):
    def __init__(self, selected_list) -> None:
        super().__init__()
        self.selected = selected_list
    def forward(self,x):
        return tuple(x[i] for i in self.selected)
    
class SelectOne(nn.Module):
    def __init__(self, i) -> None:
        super().__init__()
        self.selected = i
    def forward(self,x):
        return x[self.selected].squeeze()

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class Encoder_Decoder(nn.Module):
    def __init__(self, encoder, decoder, en_emb=None, de_emb=None) -> None:
        super().__init__()
        if en_emb is None:
            self.encoder_emb = nn.Identity()
        else:
            self.encoder_emb = en_emb
        if de_emb is None:
            self.decoder_emb = nn.Identity()
        else:
            self.decoder_emb = de_emb
        self.encoder = encoder
        self.decoder = decoder 
        
    def forward(self, x_encoder:Dict[str,torch.Tensor], x_decoder:Dict[str,torch.Tensor]=None):
        # if x.dim() == 4:
        #     batch_inputs = batch_inputs.permute(1,0,2,3)
        #     assert batch_inputs.size(1) == batch_graph.size(0) and batch_graph.dim() == 3
        #     batch_inputs = batch_inputs.flatten(start_dim= 1,end_dim=2)
        encoder_embedding = self.encoder_emb(x_encoder['data'])
        # print(x_encoder['graph'].size())
        x = self.encoder(encoder_embedding, x_encoder['graph'])
        
        
        # print(x[0].size(),x[1].size())
        if x_decoder is not None:
            x = self.decoder_emb(x,x_decoder['data'])   
        decoder_output = self.decoder(x)
        return decoder_output
    


        
    
