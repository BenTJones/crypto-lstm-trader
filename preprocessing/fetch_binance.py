import ccxt 
import pandas as pd
import time
import os
from datetime import datetime, timedelta

def fetch_binance(
    exchange_ticker = 'BTC/USDT',
    start_date = '2019-01-01T00:00:00Z',
    timeframe = '1h',
    cache_path = './data/binance_btc_usdt.csv',
    max_age_hrs = 12
    ):
    
    def path_age_check(cache_path, max_age_hrs):
        last_mod = datetime.fromtimestamp(os.path.getmtime(cache_path))
        age = datetime.now() - last_mod
        return age < timedelta(hours = max_age_hrs)
    
    if os.path.exists(cache_path) and path_age_check(cache_path,max_age_hrs):
            print(f'Loading prexisting data from {cache_path}')
            data_df = pd.read_csv(cache_path,index_col='timestamp', parse_dates= True)
            return data_df
    
    binance = ccxt.binance()
    now = binance.milliseconds()
    since = binance.parse8601(start_date) 
    data = []

    while since < now:
        ohlcv = binance.fetch_ohlcv(exchange_ticker,timeframe= timeframe, since= since , limit = 1000)
        
        if not ohlcv:
            break
        
        data += ohlcv
        
        last_time = ohlcv[-1][0] #OHLCV is a list of ms stamp,open,...,vol draws correct stamp
        since = last_time + 1 #dodge duplicate data
        
        time.sleep(binance.rateLimit / 1000) #Ensures it doesnt hit rate lim and converts to s for sleep
        
    data_df = pd.DataFrame.from_records(
        data,
        columns= ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'], unit = 'ms')
    data_df.set_index('timestamp',inplace= True)
    
    os.makedirs(os.path.dirname(cache_path), exist_ok= True)
    data_df.to_csv(cache_path)
    
    return data_df
