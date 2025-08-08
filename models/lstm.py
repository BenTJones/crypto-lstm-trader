import torch
import torch.nn as nn 

class LSTMClassifier(nn.Module):
    
    def __init__(self,input_size = 5, hidden_size = 64, num_layers = 1, dropout = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = input_size, # Determined by num of cols in features OHLCV
            hidden_size = hidden_size, #Size of hidden states,therefore memory
            num_layers = num_layers, #How many LSTM layers are stacked
            batch_first = True, #Ensures order is batch, time then features
            dropout = dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Linear(hidden_size,1)
        
    def forward(self,x):
        _, (hidden_n,_) = self.lstm(x)
        hidden_last = hidden_n[-1]
        logits = self.head(hidden_last)
        return logits.squeeze(-1)
    
    
