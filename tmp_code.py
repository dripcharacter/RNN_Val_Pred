import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class RNN_torch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_torch, self).__init__()
        self.rnn=nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc=nn.Linear(hidden_size, output_size)

    def forward(self, x_val):
        out, _ = self.rnn(x_val)
        out = self.fc(out[:, -1, :])
        return out

data=pd.read_csv("AI_Lec_23_final_stocks.csv")

def csv_to_arr(file_path):
    data=pd.read_csv(file_path)
    result=list()
    result.append(data.axes[1].values[0].split('\t'))
    for entry in range(2999):
        result.append(data.iloc[entry].values[0].split('\t'))
    result=np.array(result)
    result=result.transpose()
    return result

result=csv_to_arr("AI_Lec_23_final_stocks.csv")
print(result.shape)
seq_length = 50
X, y = [], []
print(result)
for stock_idx in tqdm(range(result.shape[0])):
    for day_idx in range(result.shape[1] - seq_length):
        X.append(result[stock_idx][day_idx:day_idx + seq_length])
        y.append(result[stock_idx][day_idx + seq_length])
        if day_idx < 3:
            print("X: ", X[-1])
            print("y: ", y[-1])


arrX=np.array(X)
arry=np.array(y)
print("X: ", arrX)
print("y: ", arry)
print("X len: ", arrX.shape)
print("y len: ", arry.shape)