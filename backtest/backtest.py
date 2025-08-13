import numpy as np

def trade_from_probs(probs,t_star,window_size,k,pre_test_wins,n_prices):
    '''Checks the probabilities vs buy condition of t star and returns potential trades
    Only trade if exit position within prices (exit < n_prices)'''
    probs = np.asarray(probs).ravel()
    trades = []
    for i,prob in enumerate(probs):
        if prob >= t_star:
            win_end = (window_size - 1) + i + pre_test_wins
            entry = win_end + 1
            exit = win_end + k
            if exit < n_prices:
                trades.append((entry,exit))
    return trades

def prevent_overlapping(trades):
    '''Returns a cleaned list to avoid skip overlapping trades'''
    cleaned = []
    last_exit = -1
    for entry,exit in trades:
        if entry > last_exit:
            cleaned.append((entry,exit))
            last_exit = exit
    return cleaned
            
def trade_returns(prices,trades,bps_fee = 20):
    '''Arithmetic return per each trade: price at exit over price at entry - 1 - fee
    bps_fee are the fees for purchase and selling in basis points 10^-2 % 20 = 0.02%'''
    
    prices = np.asarray(prices).ravel()
    if len(trades) == 0:
        print('No trade signal generated')
        return np.array([])
    
    fee = bps_fee / 10000 #Converts to decimal
    returns = []
    for entry,exit in trades:
        r = (prices[exit] / prices[entry]) - 1.0 - fee
        returns.append(r)
    return np.array(returns,dtype= float)

def equity_and_dd(returns):
    '''Equity in cumulative prod of 1+r
    Max drawdown is given by 1 - (equity / peak)'''
    
    rets = np.asarray(returns)
    if rets.size ==0:
        return [1.0],0
    equity = np.cumprod(1+rets)
    peak = np.maximum.accumulate(equity)
    dd = 1 - (equity/peak)
    max_dd = dd.max() if dd.size else 0.0
    return equity,max_dd

def sharpe_ratio(rets):
    '''Calculates the sharpe per trade: mean(ret)/std(ret).'''
    rets = np.asarray(rets)
    if rets.size <= 1:
        return float('nan')
    mu = rets.mean()
    sig = rets.std()
    sharpe = mu / sig
    return sharpe

def backtest_run(
        probs_test,
        t_star,
        price_series,
        window_size,
        k,
        global_offset,
        fee_bps=20,
        allow_overlap=False,
    ):
    prices = np.asarray(price_series).ravel()
    n_prices = len(prices)
    
    raw_trade_sigs = trade_from_probs(probs_test, t_star, window_size, k, global_offset, n_prices)
    trade_sigs = raw_trade_sigs if allow_overlap else prevent_overlapping(raw_trade_sigs)
    returns = trade_returns(prices,trade_sigs)
    equity,max_dd = equity_and_dd(returns)
    sharpe = sharpe_ratio(returns)
    
    stats = {
        "n_trades": int(returns.size),
        "hit_rate": float((returns > 0).mean()) if returns.size else float("nan"),
        "avg_ret":  float(np.mean(returns)) if returns.size else 0.0,
        "median_ret": float(np.median(returns)) if returns.size else 0.0,
        "tot_ret":  float(equity[-1] - 1.0) if returns.size else 0.0,
        "sharpe":   float(sharpe),
        "max_dd":   float(max_dd),
        "fee_bps":  int(fee_bps),
        "k":        int(k),
    }
    
    return stats, returns, trade_sigs
