import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from tqdm import tqdm


USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)

device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:',device)

data=pd.read_csv("AI_Lec_23_final_stocks.csv")

def csv_to_arr(file_path):
    data=pd.read_csv(file_path)
    result=list()
    result.append(data.axes[1].values[0].split('\t'))
    for entry in range(2999):
        result.append(data.iloc[entry].values[0].split('\t'))
    result=np.array(result)
    result=result.transpose()
    result=result.astype(np.float32)
    return result


data_df=csv_to_arr("AI_Lec_23_final_stocks.csv")


# Create sequences and labels for training
seq_length = 50
X, y = [], []
tmp=data_df.shape[1] - seq_length
for stock_idx in tqdm(range(data_df.shape[0])):
    X.append(np.array(data_df[stock_idx][tmp:tmp + seq_length]))


X = np.array(X)

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1])
        return out


input_size = 50
hidden_size = 128
output_size = 1
learning_rate = 0.001
num_epochs = 10
batch_size = 64

model=RNNModel(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('./model.pth'))

model.eval()
y_pred=[]
final_result = [[] for idx in range(100)]
with torch.no_grad():
    for idx in range(20):
        if len(y_pred)!=0:
            for sub_idx in range(len(X)):
                tmp_np=np.delete(X[sub_idx], 0)
                tmp_np=np.append(tmp_np, y_pred[sub_idx][0])
                X[sub_idx]=tmp_np
        X_test_tensor = torch.tensor(X, dtype=torch.float32)
        # print("X shape: ", X_test_tensor.shape)
        # print("X[0]: ", X_test_tensor[1])
        X_test_tensor = X_test_tensor.reshape(X_test_tensor.shape[0], X_test_tensor.shape[1], 1)
        y_pred = model(X_test_tensor).numpy()
        # print("y_pred[0]: ", y_pred[1])
        print("y_pred: {}".format(y_pred))
        print("y_pred shape: ", y_pred.shape)
        for result_idx in range(100):
            final_result[result_idx].append(y_pred[result_idx][0])

print(final_result)

