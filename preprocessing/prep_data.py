import torch 
import pandas as pd
import numpy as np

def target_def(df):
    daily_change = df['close'].shift(-1) - df['close']
    df['label'] = np.where(daily_change > 0,1,0)
    df = df[:-1]
    return df 

def target_def_knext(df,k = 4,use_log_returns = True,keep_futurek = False):
    out = df.copy()
    if use_log_returns:
        out['futurek'] = np.log(out['close'].shift(-k)) - np.log(out['close'])
    else:
        out['futurek'] = (out['close'].shift(-k) / out['close']) - 1
        
    out['label'] = np.where(out['futurek'] > 0,1.,0.)
    out = out.iloc[:-k]
    if not keep_futurek:
        out.drop(columns=['futurek'], errors='ignore', inplace=True)
    
    return out
        
def z_score_norm(df,train_frac = 0.7):
    features = df.drop(columns = ['label'])
    labels = df['label']
    
    train_end = int(features.shape[0] * train_frac)
    train_feats = features.iloc[:train_end]
    mean = train_feats.mean()
    std = train_feats.std().replace(0,1.0)
    norm_feats = (features - mean) / std
    norm_df = norm_feats.copy()
    norm_df['label'] = labels
    return norm_df

def window_creation(norm_df, window_size=48):
    features = norm_df.drop('label', axis=1)
    labels   = norm_df['label']

    X, y = [], []
    for i in range(len(labels) - window_size):
        X.append(features.iloc[i:i+window_size])
        y.append(labels.iloc[i+window_size-1])

    X = np.array(X); y = np.array(y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def train_val_test_split (x,y,train_frac = 0.7,val_frac = 0.15):
    size = len(y)
    train_end = int(size * train_frac)
    val_end = int(size * (train_frac + val_frac))
    
    x_train,y_train = x[:train_end],y[:train_end]
    x_val, y_val = x[train_end:val_end],y[train_end:val_end]
    x_test,y_test = x[val_end:],y[val_end:]
    
    return x_train, y_train, x_val, y_val, x_test, y_test

def feature_creation(df):
    out = df.copy()
    
    out['logret_close'] = np.log(df['close']).diff()
    
    for w in [12,24,48]:
        roll_close = out['close'].rolling(w, min_periods=w)
        roll_ret   = out['logret_close'].rolling(w, min_periods=w)
        roll_vol   = out['volume'].rolling(w, min_periods=w)

        out[f'close_mean{w}'] = roll_close.mean()
        out[f'close_std{w}']  = roll_close.std()
        out[f'close_min{w}']  = roll_close.min()
        out[f'close_max{w}']  = roll_close.max()

        out[f'logret_close_mean{w}'] = roll_ret.mean()
        out[f'logret_close_std{w}']  = roll_ret.std()

        vol_mean = roll_vol.mean()
        vol_std  = roll_vol.std().replace(0, np.nan)
        out[f'vol_z{w}'] = (out['volume'] - vol_mean) / vol_std

    out['rv24'] = out['logret_close'].rolling(24, min_periods=24).std()
    out['rv48'] = out['logret_close'].rolling(48, min_periods=48).std()
    
    #RSI (14 day RS) Calculation with Wilder's smoothing method
    n = 14
    change = out['close'].diff()
    gain = change.clip(lower = 0)
    loss = -change.clip(upper = 0)
    avg_gain = gain.ewm(alpha = 1/n,min_periods = n,adjust=False).mean()
    avg_loss = loss.ewm(alpha = 1/n, min_periods = n,adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0,np.nan)
    out['rsi_14'] = 100 - (100/ (1+rs))
    
    #MACD Calc and appending 
    ema_fast = out['close'].ewm(halflife = 6,adjust=False).mean()
    ema_slow = out['close'].ewm(halflife = 20,adjust=False).mean() 
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(halflife = 6,adjust = False).mean()
    out['macd'] = macd
    out['macd_signal']= macd_signal
    out['macd_hist'] = macd - macd_signal
    if hasattr(out.index, "hour"):
        hour = pd.Index(out.index).hour
        out['sin_hour'] = np.sin(2*np.pi*hour/24)
        out['cos_hour'] = np.cos(2*np.pi*hour/24)

    # Range-based vol estimators used because they have lower noise than price.
    hl_log = np.log(out['high'] / out['low']).clip(lower=0)
    oc_log = np.log(out['close'] / out['open']).fillna(0)

    # Parkinson (uses high-low)
    parkinson_var = (hl_log**2) / (4*np.log(2))
    out['parkinson_24'] = np.sqrt(parkinson_var.rolling(24, min_periods=24).mean())
    out['parkinson_48'] = np.sqrt(parkinson_var.rolling(48, min_periods=48).mean())

    # Garman–Klass (uses high, low, open, close)
    gk_var = 0.5*(hl_log**2) - (2*np.log(2)-1)*(oc_log**2)
    out['gk_24'] = np.sqrt(gk_var.rolling(24, min_periods=24).mean().clip(lower=0))
    out['gk_48'] = np.sqrt(gk_var.rolling(48, min_periods=48).mean().clip(lower=0))

    # Momentum deltas (capture acceleration of prices)
    out['drsi_14']     = out['rsi_14'].diff()
    out['dmacd_hist']  = out['macd_hist'].diff()
    
    return out.dropna()

def make_label_k_epsilon(df, k=8, eps_quantile=0.6, train_frac=0.7, use_log_returns=True):
    import numpy as np
    out = df.copy()
    r = np.log(out['close']).diff() if use_log_returns else out['close'].pct_change()
    s = r.rolling(k).sum().shift(-k)               
    n = len(s); train_end = int(n * train_frac)
    eps = float(np.nanquantile(np.abs(s.iloc[:train_end]), eps_quantile))  
    y = np.where(s >  eps, 1, np.where(s < -eps, 0, np.nan))               # drop small moves
    out['label'] = y
    out = out.iloc[:-k].dropna(subset=['label']).copy()  
    out['label'] = out['label'].astype(int)
    return out