import numpy as np
import pandas as pd
import os
import json


# df_meta = pd.read_csv('symbols_valid_meta.csv')
# tickers = df_meta['NASDAQ Symbol'].values
def get_feature(data_path, custome_tickers = None, start_date='2020-09-01', end_date = '2021-09-30'):
    if custome_tickers is None:
        ticker_path = os.path.join(data_path,'nasdaq_constituent.json')
        with open(ticker_path) as f:
            nasdaq_100 = json.load(f)
        tickers = [x['symbol'] for x in nasdaq_100]
    else:
        tickers = custome_tickers

    AAPL_path = os.path.join(data_path,'stocks_after_2020_4_01','AAPL.csv')
    ticker_df = pd.read_csv(AAPL_path )
    date_n, feature_m = ticker_df[(ticker_df['Date']>=start_date) & (ticker_df['Date']<=end_date)].shape

    ticker_list = []
    data = []
    price = []
    for ticker in tickers:
        ticker_path = os.path.join(data_path,'stocks_after_2020_4_01',ticker+'.csv')
        try:
            ticker_df = pd.read_csv(ticker_path)
        except:
            continue
        ticker_df = ticker_df[(ticker_df['Date']>=start_date) & (ticker_df['Date']<=end_date)]
        if ticker_df.shape[0]==date_n:
            tem = ticker_df.iloc[:,1:].values
            if not np.any(np.isnan(tem)):
                ticker_list.append(ticker)
                data.append(tem)
                price.append(np.array(ticker_df['Close']))
            
    data = np.stack(data).transpose(1,0,2)
    price = np.stack(price).transpose(1,0)
    return ticker_list, data, price

