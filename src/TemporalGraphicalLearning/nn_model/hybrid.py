import torch
import torch.nn as nn
import torch.nn.functional as F

class hybrid_layer(nn.Module):
    def __init__(self, graph_size, graph_model, sequence = False):
        super().__init__()
        self.sequence_flag = sequence
        self.L = nn.Parameter(torch.randn(graph_size,graph_size))

        self.graph_model = graph_model

    def forward(self, batch_inputs, batch_graphs):
        if self.sequence_flag:
            if batch_graphs.dim() == 4:
                batch_seq = batch_graphs.size()[:2]+(1,1)
            else:
                batch_seq = batch_graphs.size()[:1]+(1,1)
        else:
            if batch_graphs.dim()== 3:
                batch_seq = batch_graphs.size()[:1]+(1,1)
            else:
                batch_seq = (1,1)
        L = F.relu(self.L-0.5)
        L = L.repeat(batch_seq)
        batch_graphs += L
        out_put = self.graph_model(batch_inputs, batch_graphs)
        return out_put
    
class dynamic_layer(nn.Module):
    def __init__(self, graph_size, graph_model, sequence = False):
        super().__init__()
        self.sequence_flag = sequence
        self.L = nn.Parameter(torch.randn(graph_size,graph_size))

        self.graph_model = graph_model

    def forward(self, batch_inputs, batch_graphs):
        if self.sequence_flag:
            if batch_graphs.dim() == 4:
                batch_seq = batch_graphs.size()[:2]+(1,1)
            else:
                batch_seq = batch_graphs.size()[:1]+(1,1)
        else:
            if batch_graphs.dim()== 3:
                batch_seq = batch_graphs.size()[:1]+(1,1)
            else:
                batch_seq = (1,1)
        
        L = F.relu(self.L-2)
        L = L.repeat(batch_seq)
        batch_graphs = L
        out_put = self.graph_model(batch_inputs, batch_graphs)
        return out_put