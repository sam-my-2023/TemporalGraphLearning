import torch.nn as nn

class Node_MultiheadAttention(nn.Module):
    def __init__(self, multiheadAttention_module):
        super().__init__()
        self.module = multiheadAttention_module
    
    def forward(self, x):
        q, k, v = x
        q = q.permute(1,0,2,3)
        q = q.flatten(start_dim=1,end_dim = 2)
        
        k = k.permute(1,0,2,3)
        
        v = v.permute(1,0,2,3)
        v = v.flatten(start_dim=1,end_dim = 2)
        output = self.module(q,k,v)
        output = output.unflatten()
        return output
    