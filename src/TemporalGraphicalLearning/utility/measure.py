import numpy as np 
from scipy.stats import spearmanr
def simple_denoise(contiguous_price: np.ndarray):
    n = len(contiguous_price)

    ret = contiguous_price[1:]/contiguous_price[:-1]
    average_ret = np.mean(ret, axis=1)
    ret = ret - average_ret[:,None]
    return ret

def emprical_alpha_beta(contiguous_price,r=10,l = 30, implied_benmark = None):
    
    n, m = contiguous_price.shape
    if implied_benmark is None:
        implied_benmark= np.ones(m)/m
    benchmark_price = contiguous_price@implied_benmark

    benchmark_ret_move = np.stack([benchmark_price[i+l-r:i+l+1+r]/benchmark_price[i+l] for i in range(r,n-r-l)])
    ret_move = np.stack([contiguous_price[i+l-r:i+l+1+r]/benchmark_price[i+l] for i in range(r,n-r-l)])
    beta = np.array([[spearmanr(ret_move[j,:,i],benchmark_ret_move[j]).correlation for i in range(m) ] for j in range(n-2*r-l)])
    delta = np.array([[np.mean([ret_move[j,:,i]-benchmark_ret_move[j]]) for i in range(m) ] for j in range(n-2*r-l)])

    return delta,beta

def tag_emprical_alpha_beta(contiguous_price,r=10,l = 30, implied_benmark = None):
    
    delta, beta = emprical_alpha_beta(contiguous_price,r,l,implied_benmark)
    
    #            delta > 0, detla <= 0
    # beta > 0  :        3,      1,
    # beta <= 0 :        2,      0,
    tag = 2*(delta>0)+1*(beta>0)
    n_classes = 4
    return tag, n_classes

def MS_daily_mu_sigma(contiguous_price):
    return np.mean(contiguous_price,axis=0),np.var(contiguous_price,axis=0)

def sMAPE_np(predicted, contiguous_price):
    ''' symmetric mean absolute percentage error
    2/h (\sum^{n+h}_{t=n+1} \frac{|Y_t - \cut{Y}_t|}{|Y_t| + |\cut{Y}_t|})* 100%

    '''

    # checek length
    h = predicted.size[0]
    assert h== contiguous_price.size[0]
    tem = np.abs(contiguous_price-predicted)/(np.abs(contiguous_price)+np.abs(predicted))
    tem = 2*np.sum(tem)/h
    return tem

def MASE_np(predicted, contiguous_price, m):
    ''' mean absolute scaled error
    1/h \frac{\sum_{}^{} |Y_t- \cut{Y}_t|}{\frac{1}{n-m} \sum |Y_t - Y_{t-m}|}
    '''
    h = predicted.size[0]
    n = contiguous_price.size[0] - h
    assert m<n
    
    tem = np.sum(np.abs(contiguous_price[-h:]-predicted))
    tem = tem/np.sum(np.abs(contiguous_price[m:n]-contiguous_price[0:n-m]))
    tem = tem/h*(n-m)
    return tem
    


def n_later_days_distribution(contiguous_price: np.ndarray, post_l:int = 15, step_days = 1,reduce = "mean", breakpoints= [ 1.1,0.9]):
    n_seq = len(contiguous_price)
    if reduce == 'mean':
        ret = np.stack([np.stack([np.mean(contiguous_price[i+1:i+post_l+1]/contiguous_price[i]>= bp,axis=0) for bp in breakpoints], axis=1)  for i in range(n_seq-post_l)])
        return ret
    elif reduce == 'sort':
        m = contiguous_price.shape[1]
        num_classes  = 3
        ret = np.stack([np.mean(contiguous_price[i+step_days:i+(post_l+1)*step_days:step_days],axis=0)/contiguous_price[i]  for i in range(n_seq-step_days*post_l)])
        k = m//num_classes 
        sort_idx = np.argsort(ret, axis=1)
        tag = np.zeros(sort_idx.shape)
        n = tag.shape[0]
        for i in range(n):
            tag[i,sort_idx[i,k:2*k]] = 1
            tag[i,sort_idx[i,2*k:m]] = 2
        return tag, num_classes

def equi_sort(contiguous_price: np.ndarray):
    m = contiguous_price.shape[1]
    ret = contiguous_price[1:]/contiguous_price[:-1]
    n_classes =3 
    k = m//n_classes
    sort_idx = np.argsort(ret, axis=1)
    tag = np.zeros(sort_idx.shape)
    n = tag.shape[0]
    for i in range(n):
        tag[i,sort_idx[i,k:2*k]] = 1
        tag[i,sort_idx[i,2*k:m]] = 2
    return tag,n_classes
    

if __name__=="__main__":
    a = np.array([[1,1,3],[2,1,1], [3,1,2],[4,2,1]])
    a = n_later_days_distribution(a,n=2)
    print(a[1][1])
    

def realized_variance(contiguous_price: np.ndarray, k = 1, m = 1):
    n_seq = contiguous_price.shape[0]
    assert k%m == 0
    lag = k//m
    R = (contiguous_price[lag:] - contiguous_price[:-lag])/contiguous_price[:-lag]
    n_seq = n_seq
    R_square = R*R
    rv = np.stack([np.sum(R_square[i-k:i:lag],axis=0) for i in range(k,n_seq)])

    return rv, n_seq-lag