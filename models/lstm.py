import torch
import torch.nn as nn 

class LSTMClassifier(nn.Module):
    
    def __init__(self, input_size, hidden_size=64, num_layers=1, head_dropout=0.2, input_dropout=0.1):
        super().__init__()
        self.in_drop = nn.Dropout(input_dropout)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0  # only works if num_layers > 1
        )
        self.head_drop = nn.Dropout(head_dropout)
        self.head = nn.Linear(hidden_size, 1)
        
    def forward(self,x):
        _, (hidden_n,_) = self.lstm(x)
        hidden_last = hidden_n[-1]
        logits = self.head(hidden_last)
        return logits.squeeze(-1)
    
    
