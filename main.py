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
tmp=data_df.shape[1] - seq_length-1
for stock_idx in tqdm(range(data_df.shape[0])):
    X.append(np.array(data_df[stock_idx][tmp:tmp + seq_length]))
    y.append(np.array(data_df[stock_idx][tmp + seq_length]))

X, y = np.array(X), np.array(y)

# Split the data into training and test sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)

# Create a custom dataset class for PyTorch DataLoader
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.X = self.X.reshape(self.X.shape[0], self.X.shape[1], 1)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.y = self.y.reshape(self.y.shape[0], 1)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1])
        return out

# Hyperparameters
input_size = seq_length
hidden_size = 128
output_size = 1
learning_rate = 0.001
num_epochs = 10
batch_size = 64

# Create data loaders
train_dataset = TimeSeriesDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size)

# Initialize the model, loss function, and optimizer
model = RNNModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Training the model
for epoch in tqdm(range(num_epochs)):
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation on the test set
model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    X_test_tensor = X_test_tensor.reshape(X_test_tensor.shape[0], X_test_tensor.shape[1], 1)
    y_pred = model(X_test_tensor).numpy()

# Calculate RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

try:
    torch.save(model.state_dict(), './model.pth')
except:
    print("failed to save model")
else:
    print("successed to save model as ./model.pth")

# get real prediction values(3001~3020)
X = []
tmp=data_df.shape[1] - seq_length
for stock_idx in tqdm(range(data_df.shape[0])):
    X.append(np.array(data_df[stock_idx][tmp:tmp + seq_length]))
X = np.array(X)

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
        X_test_tensor = X_test_tensor.reshape(X_test_tensor.shape[0], X_test_tensor.shape[1], 1)
        y_pred = model(X_test_tensor).numpy()
        for result_idx in range(100):
            final_result[result_idx].append(y_pred[result_idx][0])

print("final result: {}".format(final_result))