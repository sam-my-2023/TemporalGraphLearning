import numpy as np 
from scipy.stats import spearmanr

def label_sort(contiguous_price: np.ndarray, num_classes = 3, axis = 0):
    if axis==1:
        contiguous_price = contiguous_price.T
    n,m = contiguous_price.shape
    k = n//num_classes 
    sort_idx = np.argsort(contiguous_price, axis=1)
    tag = np.zeros(sort_idx.shape)
    for i in range(m):
        tag[sort_idx[k:m-k,i],i] = 1
        tag[sort_idx[m-k:m,i],i] = 2
    return tag, num_classes

def return_rate(contiguous_price: np.ndarray, lag:int = 15, step_days:int = 1):
    n_seq = len(contiguous_price)
    ret = np.stack([contiguous_price[i+step_days:i+(lag+1)*step_days:step_days]/contiguous_price[i]  for i in range(n_seq-step_days*lag)])
    
    ret = ret.transpose(0,2,1)
    return (ret-1)*100

def lag_distribution(contiguous_price: np.ndarray, lag:int = 15, step_days:int = 1,reduce:str = "sort", reduce_arg:list[int]=[0.3,0.4,0.3]):
    n_seq = len(contiguous_price)
    if reduce == 'mean':
        raise
        ret = np.stack([np.stack([np.mean(contiguous_price[i+1:i+lag+1]/contiguous_price[i]>= bp,axis=0) for bp in breakpoints], axis=1)  for i in range(n_seq-lag)])
        return ret
    elif reduce == 'sort':
        assert sum(reduce_arg) ==1
        num_classes = len(reduce_arg)
        m = contiguous_price.shape[1]
        ret = np.stack([np.mean(contiguous_price[i+step_days:i+(lag+1)*step_days:step_days],axis=0)/contiguous_price[i]  for i in range(n_seq-step_days*lag)])
        k = m
        reduce_arg = [ sum(reduce_arg[:_+1]) for _ in range(num_classes)]
        reduce_arg = [ int(_*m) for _ in reduce_arg]
        sort_idx = np.argsort(ret, axis=1)
        tag = np.zeros_like(sort_idx)
        n = tag.shape[0]
        for i in range(n):
            for k in range(1,num_classes):
                tag[i,sort_idx[i,reduce_arg[k-1]:reduce_arg[k]]] = k
        return tag, num_classes