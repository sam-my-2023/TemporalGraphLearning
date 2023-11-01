import torch
import torch.nn as nn
import torch.nn.functional as F

class multi_graph_attention_layer(nn.Module):
    def __init__(self, num_graphs,graph_size, graph_model, out_layer, sequence = False):
        super().__init__()
        self.sequence_flag = sequence
        self.multi_L = nn.Parameter(torch.randn(num_graphs, graph_size,graph_size))

        self.graph_model = graph_model
        self.out_layer = out_layer

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
        
        multi_L = F.relu(self.multi_L-2)
        vs = []
        qs = []
        for i in range(multi_L.size(0)):
            single_L = multi_L[i]
            batch_seq_single_L = single_L.repeat(batch_seq)
            v,q = self.graph_model(batch_inputs, batch_seq_single_L)
            vs.append(v)
            qs.append(q)
        vs = torch.stack(vs)
        qs = torch.stack(qs)
        out_put = torch.bmm(vs,vs)@qs
        out_put = self.out_layer(out_put)
        return out_put
    
# class multi_graph_model(nn.Module):
#     def __init__(self, graph_model, sequence = False):
#         super().__init__()
#         self.sequence_flag = sequence


#     def forward(self, batch_inputs, multi_graphs):

#         if multi_graphs.dim() == 3+self.sequence_flag:
#             # case for non_batched graph
#             # suppose 
#             #   multi_graphs ~ (n_graphs, [seq_length,] num_companies,num_companies)
#             #   input ~ (seq_length, num_companies, feature_size)
#             #  
#             batch_inputs = batch_inputs.repeat(*[1]*(1+self.sequence_flag) ,multi_graphs.size(0),1)
#             multi_graphs,_ =  
#         elif multi_graphs.dim() == 4+self.sequence_flag:
#             # case for batched data
#             #   multi_graphs ~ (batch,sequence, , num_companies,num_companies)
#             #   input ~ (seq, num_companies, feature_size)
            
