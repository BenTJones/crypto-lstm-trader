import torch 
import pandas as pd
import numpy as np

def target_def(df):
    daily_change = df['close'].shift(-1) - df['close']
    df['label'] = np.where(daily_change > 0,1,0)
    df = df[:-1]
    return df 

def z_score_norm(df,split_time = '2023-01-01 00:00:00'):
    norm_df = df.drop('label',axis = 1)
    test = norm_df.loc[:'2023-01-01 00:00:00']
    mean = test.mean()
    std = test.std()
    norm_df = (norm_df - mean) / std
    norm_df['label'] = df['label']
    return norm_df

def window_creation(norm_df,window_size = 48):
    features = norm_df.drop('label',axis = 1)
    labels = norm_df['label']
    
    x = []
    y = []
    for i in range(len(labels)-window_size):
        x.append(features[i:i + window_size])
        y.append(labels[i+window_size])
        
    x = np.array(x)
    y = np.array(y)
    
    x_tensor = torch.tensor(x, dtype= torch.float32)
    y_tensor = torch.tensor(y, dtype= torch.float32)

    return x_tensor,y_tensor

def train_val_test_split (x,y,train_frac = 0.7,val_frac = 0.15):
    size = len(y)
    train_end = int(size * train_frac)
    val_end = int(size * (train_frac + val_frac))
    
    x_train,y_train = x[:train_end],y[:train_end]
    x_val, y_val = x[train_end:val_end],y[train_end:val_end]
    x_test,y_test = x[val_end:],y[val_end:]
    
    return x_train, y_train, x_val, y_val, x_test, y_test

