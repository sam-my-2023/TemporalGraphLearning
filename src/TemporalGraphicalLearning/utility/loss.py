import numpy as np 
from scipy.stats import spearmanr
import torch
from typing import Union
import torch.nn as nn
from torch.nn.functional import relu
from torch.nn import MSELoss

np_tensor = Union[torch.Tensor,np.array]

class QuantileLoss(nn.Module):
    def __init__(self, quantiles:np_tensor):
            super().__init__()
            self.quantiles = quantiles
            if isinstance(quantiles,torch.Tensor) and quantiles.device.type=='cuda':
                quantiles = quantiles.to('cpu').detach().numpy()
                self.np_quantiles = quantiles
            mid_idx = np.argwhere(quantiles==0.5)
            assert len(mid_idx)==1
            self.mid_idx = mid_idx[0]
        
    def forward(self, preds:torch.Tensor, target:torch.Tensor):
        # check batch_size
        
        n_predicted_quantiles = preds.size(-1)
        n_target_samples = target.size(-1)
        target = target.unsqueeze(-1).repeat([1]*target.dim()+[n_predicted_quantiles])
        errors = target - preds.unsqueeze(-2).expand(target.size())
        loss = relu(-errors)@(1-self.quantiles) + relu(errors)@self.quantiles
        loss = torch.sum(loss)
        return loss
    
    @torch.no_grad()
    def eval_quantile(self,our_model,test_ds, quantile= 0.5):
        idx = np.argwhere(self.np_quantiles==quantile)
        assert len(idx)==1
        idx = idx[0]
        pred = []
        ground_truth = []
        
        for seq_stock,tag,seq_graph in test_ds:
            logits = our_model(seq_stock,seq_graph)
            pred.append(logits.to('cpu').detach().numpy())
            ground_truth.append(tag.to('cpu').detach().numpy())
        pred = np.concatenate(pred)
        ground_truth = np.concatenate(ground_truth)

        pred = pred[...,idx]
        k_th = int(quantile*ground_truth.shape[1])
        # np.partition will be faster than sort , partition in place
        ground_truth.partition(k_th) 
        ground_truth = ground_truth[...,k_th]
        return pred,ground_truth 
        
    @torch.no_grad()
    def eval_quantile_x(self,our_model, test_ds, quantile= 0.5):
        idx = np.argwhere(self.np_quantiles==quantile)
        assert len(idx)==1
        idx = idx[0]
            
        
        pred = []
        ground_truth = []
        
        for x in test_ds:
            x = {key:x[i]for i,key in enumerate(test_ds.keys) }
            logits = our_model(x)
            pred.append(logits.to('cpu').detach().numpy())
            ground_truth.append(x['tag'].to('cpu').detach().numpy())
        pred = np.concatenate(pred)
        ground_truth = np.concatenate(ground_truth)

        pred = pred[...,idx]
        k_th = int(quantile*ground_truth.shape[1])
        # np.partition will be faster than sort , partition in place
        ground_truth.partition(k_th) 
        ground_truth = ground_truth[...,k_th]
        return pred,ground_truth
    
    # def as_measure(self, pred,ground_truth, quantile):
        # pred,ground_truth = loss_nn.eval_quantile_x(model,dataset)
        # pg_pairs = np.stack([pred.flatten(),ground_truth],axis=-1)
        # # print(pg_pairs[0:960:96],pg_pairs[1:961:96])
        # # print(pred.shape,ground_truth.shape)
        # scores_ = {
        #             'MAE':mean_absolute_error(ground_truth,pred),
        #             # 'macro F1': f1_score(ground_truth,pred, average = 'macro'),
        #             # 'micro F1': f1_score(ground_truth,pred, average = 'micro')
        #             }
        # global nn_parameters
        # nn_parameters = model
        
        # return scores_['MAE']