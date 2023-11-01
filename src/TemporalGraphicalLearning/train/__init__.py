from collections import defaultdict
import json
import os
import numpy as np
# from .nn_model.gcnn_lstm import Eval_Dataset


class From_Jason_File:
    '''
    a wrapper for data read from chatgpt jason
    '''
    def __init__(self, nest_list_data) -> None:
        
        # dtype as np.int64 corresponds to pandas df wrds 
        self.dates = [np.int64(e['Date'].replace('-', '')) for e in nest_list_data]
        
        self.affected_companies = [e['Affected Companies'] for e in nest_list_data]
        self._process()
        
    def _process(self):
        total_companies =  [company for daily_afftected in self.affected_companies for company in list(daily_afftected.keys())]
        self.total_companies = list(set(total_companies))
        num_dates = len(self.dates)
        relation = []
        for i in range(num_dates):
            tem = defaultdict(lambda: [])
            for company,flag in self.affected_companies[i].items():
                tem[flag].append(company)
            relation.append(tem)
            
        
        self.relation = relation

    def __len__(self):
        return len(self.dates)
    
    def get_graphs(self,  relation_type = "positive"):
        return [x[relation_type] for x in self.relation]
    
    def get_dates(self):
        return self.dates
    
    def get_graph_data(self, date_idx=0, date = None, relation_type = "positive"):
        if date is not None:
            print(date)
        else:
            return self.relation[date_idx][relation_type]

def adjacent_matrix(positive_rel,negative_rel,TICKERS_IND, relation_types = ['positive']):
    m = len(TICKERS_IND)
    tA = np.zeros([m,m])
    r = len(positive_rel)
    assert r==len(negative_rel)

 # positive
    if 'positive' in relation_types:
        for graph in positive_rel:
            A = np.zeros([m,m])
            
            for k in range(len(graph)):
                i = TICKERS_IND[graph[k]]
                for c in graph[k+1:]:
                    j = TICKERS_IND[c]
                    tA[i,j] = 1
                    tA[j,i] = 1

 # negative set

    if 'negative' in relation_types:
        for graph in negative_rel:
            A = np.zeros([m,m])
            
            for k in range(len(graph)):
                i = TICKERS_IND[graph[k]]
                for c in graph[k+1:]:
                    j = TICKERS_IND[c]
                    tA[i,j] = 1
                    tA[j,i] = 1

    return tA



