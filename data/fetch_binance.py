import ccxt 
import pandas as pd
import time

def fetch_binance(exchange_ticker = 'BTC/USDT',start_date = '2019-01-01T00:00:00Z',timeframe = '1h'):
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
        
        time.sleep(binance.rateLimit / 1000)
        
    data_df = pd.DataFrame.from_records(
        data,
        columns= ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
    data_df.set_index('timestamp',inplace= True)

    return data_df
    